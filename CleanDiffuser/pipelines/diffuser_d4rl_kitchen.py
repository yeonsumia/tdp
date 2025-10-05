import os

import d4rl
import gym
import hydra
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import os
from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.d4rl_kitchen_dataset import D4RLKitchenDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters
from utils import set_seed

from diffusion.tree.tree import TrajAggTree

OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
    }

BONUS_THRESH = 0.3

# kitchen-mixed-v0
TASK_ELEMENTS = ["kettle", "microwave", "light switch", "bottom burner"]

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

@hydra.main(config_path="../configs/diffuser/kitchen", config_name="kitchen", version_base=None)
def pipeline(args):

    set_seed(args.seed)

    save_path = f'CleanDiffuser/results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    dataset = D4RLKitchenDataset(
        env.get_dataset(), horizon=args.task.horizon, discount=args.discount)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # --------------- Network Architecture -----------------
    nn_diffusion = JannerUNet1d(
        obs_dim + act_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", attention=False, kernel_size=5)
    nn_classifier = HalfJannerUNet1d(
        args.task.horizon, obs_dim + act_dim, out_dim=1,
        model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.task.dim_mult,
        timestep_emb_type="positional", kernel_size=3)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"======================= Parameter Report of Classifier =======================")
    report_parameters(nn_classifier)
    print(f"==============================================================================")

    # --------------- Classifier Guidance --------------------
    classifier = CumRewClassifier(nn_classifier, device=args.device)

    # ----------------- Masking -------------------
    fix_mask = torch.zeros((args.task.horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.
    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = args.action_loss_weight

    # --------------- Diffusion Model --------------------
    agent = DiscreteDiffusionSDE(
        nn_diffusion, None,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, ema_rate=args.ema_rate,
        device=args.device, diffusion_steps=args.diffusion_steps, predict_noise=args.predict_noise)

    # ---------------------- Training ----------------------
    if args.mode == "train":

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
        classifier_lr_scheduler = CosineAnnealingLR(agent.classifier.optim, args.classifier_gradient_steps)

        agent.train()

        n_gradient_step = 0
        log = {"avg_loss_diffusion": 0., "avg_loss_classifier": 0.}

        for batch in loop_dataloader(dataloader):

            obs = batch["obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            val = batch["val"].to(args.device)

            x = torch.cat([obs, act], -1)

            # ----------- Gradient Step ------------
            log["avg_loss_diffusion"] += agent.update(x)['loss']
            diffusion_lr_scheduler.step()
            if n_gradient_step <= args.classifier_gradient_steps:
                log["avg_loss_classifier"] += agent.update_classifier(x, val)['loss']
                classifier_lr_scheduler.step()

            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= args.log_interval
                log["avg_loss_classifier"] /= args.log_interval
                print(log)
                log = {"avg_loss_diffusion": 0., "avg_loss_classifier": 0.}

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                agent.classifier.save(save_path + f"classifier_ckpt_{n_gradient_step + 1}.pt")
                agent.save(save_path + f"diffusion_ckpt_latest.pt")
                agent.classifier.save(save_path + f"classifier_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break

    # ---------------------- Inference ----------------------
    elif args.mode == "inference":

        agent.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        agent.classifier.load(save_path + f"classifier_ckpt_{args.ckpt}.pt")

        agent.eval()

        env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer = dataset.get_normalizer()
        episode_rewards = []

        prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)
        for i in range(args.num_episodes):
            # actual_tasks_to_complete = dict()
            # for i in range(args.num_envs):
            #     actual_tasks_to_complete[i] = TASK_ELEMENTS.copy()
            if args.use_tree:
                traj_agg_tree_lst = [TrajAggTree(tree_lambda=args.tree_lambda, 
                                            traj_dim=obs_dim+act_dim,
                                            one_minus_alpha=args.one_minus_alpha,
                                            ) for _ in range(args.num_envs)]

            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

            while not np.all(cum_done) and t < 280 + 1:
                # normalize obs
                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

                # sample trajectories
                # print(f"prior: {prior.shape}, obs: {obs_dim}, act: {act_dim}")
                # obs: 60 / act: 9
                # obs (total 60): robot (9 (N_DOF_ROBOT)), object (21 (N_DOF_OBJECT)), goal (30)
                # print(f"cg: {args.task.w_cg}")
                # print(f"obs: {obs.shape}") 
                prior[:, 0, :obs_dim] = obs # shape: [num_envs, obs_dim+act_dim]
                traj, log = agent.sample(
                    prior.repeat(args.num_candidates, 1, 1),
                    solver=args.solver,
                    n_samples=args.num_candidates * args.num_envs,
                    sample_steps=args.sampling_steps,
                    use_ema=args.use_ema, w_cg=args.task.w_cg, temperature=args.temperature)
                
                # Tree
                if args.use_tree:
                    traj_obs = traj.view(args.num_candidates, args.num_envs, args.task.horizon, -1)
                    act_lst = []
                    for et in range(args.num_envs):
                        # Aggregate
                        traj_agg_tree_lst[et].integrate_trajectories(traj_obs[:, et, :, :].detach().cpu().numpy())
                        # Acting 
                        next_sample, selected_key, _, _ = traj_agg_tree_lst[et].get_next_state_original()
                        # Pruning 
                        traj_agg_tree_lst[et].pruning(selected_key)
                        # print(next_sample.shape)
                        act_lst.append(next_sample[obs_dim:])
                    act = np.array(act_lst).clip(-1., 1.)
                # calculate distance
                # diff = traj[:100].view(100, args.task.horizon, -1)[..., :17].unsqueeze(1) - traj[:100].view(100, args.task.horizon, -1)[..., :17].unsqueeze(0)
                # remove the diag, make distance with shape [N * N-1 * H * D]
                # diff = diff[~torch.eye(diff.shape[0], device=traj.device).bool()].reshape(diff.shape[0], -1, *diff.shape[2:])
                # distance
                # print(f"diff: {diff.shape}")
                # distance = torch.norm(diff, p=2, dim=(-2, -1), keepdim=True)

                # # mean distance
                # distance = distance.mean(dim=1, keepdim=True)
                # distance = sum(distance.squeeze(0)) / distance.shape[0]
                # print(f"Distance: {distance}")
                # select the best plan
                elif args.use_sub_tree:
                    traj_obs = traj.view(args.num_candidates, args.num_envs, args.task.horizon, -1)
                    
                    # sample sub-traj
                    a = 4
                    possible_indices = torch.arange(16, args.task.horizon, step=a)  # Indices: {a, 2a, ..., h-1}
                    # sub_plan_index = possible_indices[torch.randperm(len(possible_indices))[:8]]
                    # sub_plan_index, _ = torch.sort(sub_plan_index)
                    # sub_plan_index = sub_plan_index.repeat(args.batch_size // 8)
                    # random sampling index for each args.num_cand * args.num_envs in possible_indices
                    sub_plan_index = possible_indices[torch.randint(0, len(possible_indices), (args.num_candidates, args.num_envs,))]
                    print(sub_plan_index)
                    sub_prior = torch.zeros((args.num_candidates, args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)
                    # set sub_prior with obs and sub_plan_index
                    for i in range(args.num_candidates):
                        for j in range(args.num_envs):
                            sub_prior[i, j, 0, :obs_dim] = traj_obs[i, j, sub_plan_index[i, j], :obs_dim]
                    
                    sub_prior = sub_prior.view(args.num_candidates * args.num_envs, args.task.horizon, obs_dim + act_dim)
                    
                    # sample sub trajectories
                    sub_traj, sub_log = agent.sample(
                        sub_prior,
                        solver=args.solver,
                        n_samples=args.num_candidates * args.num_envs,
                        sample_steps=args.sampling_steps,
                        use_ema=args.use_ema, w_cg=0.0005, temperature=args.temperature)

                    # evaluate sub trajectories
                    merged_traj = []
                    for i in range(args.num_candidates * args.num_envs):
                        # merge sub_traj and traj
                        merged_traj.append(traj[i])
                        # print(traj[i, :sub_plan_index[i//args.num_envs, i%args.num_envs], :].shape, sub_traj[i, :, :].shape)
                        merged_traj.append(torch.cat([traj[i, :sub_plan_index[i//args.num_envs, i%args.num_envs], :], sub_traj[i, :, :]], dim=0))
                    tasks_to_complete_total = dict()
                    for i in range(args.num_envs * args.num_candidates * 2):
                        tasks_to_complete_total[i] = TASK_ELEMENTS.copy()
                        # print(f"Env [{i}]: {tasks_to_complete_total[i]}")
                    score_lst = calculate_sequence_score_sub_traj(tasks_to_complete_total, merged_traj)
                    print(f"score_lst: {len(score_lst)}") # Total number of trajectories (num_envs * num_candidates * 2)
                    score_lst = np.array(score_lst).reshape(args.num_candidates, args.num_envs, 2).transpose((0, 2, 1)).reshape(args.num_candidates * 2, args.num_envs)
                    idx = score_lst.argmax(0)
                    logp = log["log_p"].view(args.num_candidates, args.num_envs, -1).sum(-1)
                    for i, idx_ in enumerate(idx):
                        if score_lst[idx_, i] == 0:
                            # print(f"here: {logp[:, i].argmax(0)}")
                            idx[i] = logp[:, i].argmax(0) * 2
                    for i in range(args.num_envs):
                        print(f"Env {i} max score: {score_lst[idx[i], i]}")
                    act = traj.view(args.num_candidates, args.num_envs, args.task.horizon, -1)[
                        idx // 2, torch.arange(args.num_envs), :2, obs_dim:]
                    act = act.clip(-1., 1.).cpu().numpy()
                else:
                    logp = log["log_p"].view(args.num_candidates, args.num_envs, -1).sum(-1)
                    # agent is only provided information about total score
                    tasks_to_complete_total = dict()
                    for i in range(args.num_envs * args.num_candidates):
                        tasks_to_complete_total[i] = TASK_ELEMENTS.copy()
                    score_lst = calculate_sequence_score(tasks_to_complete_total, traj)
                    # print(f"score_lst: {score_lst}")
                    score_lst = np.array(score_lst).reshape(args.num_candidates, args.num_envs)
                    idx = score_lst.argmax(0)
                    # print(f"idx: {idx.shape}") # shape: (args.num_envs,)
                    for i, idx_ in enumerate(idx):
                        if score_lst[idx_, i] == 0:
                            # print(f"here: {logp[:, i].argmax(0)}")
                            idx[i] = logp[:, i].argmax(0)
                    for i in range(args.num_envs):
                        print(f"Env {i} max score: {score_lst[idx[i], i]}")
                    act = traj.view(args.num_candidates, args.num_envs, args.task.horizon, -1)[
                        idx, torch.arange(args.num_envs), 0, obs_dim:]
                    act = act.clip(-1., 1.).cpu().numpy()

                # step
                obs, rew, done, info = env_eval.step(act[:, 0, :])
                obs, rew, done, info = env_eval.step(act[:, 1, :])
                # print(info[0]['obs_dict']['obj_qp'].shape)
                # obs_ = np.expand_dims(obs, axis=1)
                # rew = calculate_sequence_score_np(actual_tasks_to_complete, obs[..., :obs_dim])
                # # print(f"rew: {rew}")
                # rew = np.asarray(rew)

                t += 2
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += rew
                if t % 10 == 0:
                    # for i in range(args.num_envs):
                    #     print(f"Env [{i}]: {actual_tasks_to_complete[i]}")
                    print(f'[t={t}] cum_rew: {ep_reward}, ')
                        # f'logp: {logp[idx, torch.arange(args.num_envs)]}')

            # clip the reward to [0, 4] since the max cumulative reward is 4
            episode_rewards.append(np.clip(ep_reward, 0., 4.))

        episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
        episode_rewards = np.array(episode_rewards)
        print(np.mean(episode_rewards, -1), np.std(episode_rewards, -1))

    else:
        raise ValueError(f"Invalid mode: {args.mode}")

def calculate_sequence_score(tasks_to_complete, traj):
    # tasks_completed: {num_envs * num_candidates: list}
    # traj: [num_envs * num_candidates, planning_horizon, obs_dim+act_dim]
    # succeed but wrong order :  +0.5
    # succeed in correct order:  +1.0
    # failed                  :  +0.0
    # tasks_to_complete = TASK_ELEMENTS
    # tasks_completed = []
    # planning_horizon: 32 -> total planning steps: 280 (closed-loop planning)
    # print(f"traj: {traj.shape}") [num_envs * num_candidates, planning_horizon, obs_dim+act_dim]
    # output: score: [num_envs * num_candidates]
    score_lst = [0 for _ in range(traj.shape[0])]
    for i, state in enumerate(traj):
        # print(state.shape)
        for t in range(traj.shape[1]):
            qs = state[t, :9].detach().cpu().numpy()
            q_objs = state[t, 9:30].detach().cpu().numpy()
            goal = state[0][30:].detach().cpu().numpy()
            score, completions = _get_reward_n_score(tasks_to_complete[i], qs, q_objs, goal)
            for e in completions:
                # print(f"complete: {e}")
                tasks_to_complete[i].remove(e)
            score_lst[i] += score
            # if score > 0:
            #     print(f"[{t}/{traj.shape[1]}] score: {score}, completions: {completions}")
    # print(score_lst)
    return score_lst


def calculate_sequence_score_sub_traj(tasks_to_complete, sub_traj):
    # tasks_completed: {num_envs * num_candidates: list}
    # traj: [num_envs * num_candidates, planning_horizon, obs_dim+act_dim]
    # succeed but wrong order :  +0.5
    # succeed in correct order:  +1.0
    # failed                  :  +0.0
    # tasks_to_complete = TASK_ELEMENTS
    # tasks_completed = []
    # planning_horizon: 32 -> total planning steps: 280 (closed-loop planning)
    # print(f"traj: {traj.shape}") [num_envs * num_candidates, planning_horizon, obs_dim+act_dim]
    # output: score: [num_envs * num_candidates]
    score_lst = [0 for _ in range(len(sub_traj))]
    for i, state in enumerate(sub_traj):
        # print(state.shape)
        for t in range(len(state)):
            qs = state[t, :9].detach().cpu().numpy()
            q_objs = state[t, 9:30].detach().cpu().numpy()
            goal = state[t, 30:].detach().cpu().numpy()
            score, completions = _get_reward_n_score(tasks_to_complete[i], qs, q_objs, goal)
            for e in completions:
                # print(f"complete: {e}")
                tasks_to_complete[i].remove(e)
            score_lst[i] += score
            # if score > 0:
            #     print(f"[{t}/{traj.shape[1]}] score: {score}, completions: {completions}")
    # print(score_lst)
    return score_lst

def calculate_sequence_score_np(tasks_to_complete, traj):
    # tasks_completed: {num_envs * num_candidates: list}
    # traj: [num_envs * num_candidates, planning_horizon, obs_dim+act_dim]
    # succeed but wrong order :  +0.5
    # succeed in correct order:  +1.0
    # failed                  :  +0.0
    # tasks_to_complete = TASK_ELEMENTS
    # tasks_completed = []
    # planning_horizon: 32 -> total planning steps: 280 (closed-loop planning)
    # print(f"traj: {traj.shape}") [num_envs * num_candidates, planning_horizon, obs_dim+act_dim]
    score_lst = []
    for i, state in enumerate(traj):
        qs = state[:9]
        q_objs = state[9:30]
        goal = state[30:]
        # print(qs.shape, goal.shape)
        # print(f"Env [{i}]: {tasks_to_complete[i]}")
        score, completions = _get_reward_n_score(tasks_to_complete[i], qs, q_objs, goal)
        # print(completions)
        for e in completions:
            print(f"complete: {e}")
            # print(f"before: {tasks_to_complete[i]}")
            tasks_to_complete[i].remove(e)
            # print(f"after: {tasks_to_complete[i]}")
        score_lst.append(sum(score) if len(score) > 0 else 0)
        # print(f"score: {score}")
    
    return score_lst
        

def _get_reward_n_score(tasks_to_complete_traj, q_obs, q_obj_obs, goal):
        # reward_dict, score = super(KitchenBase, self)._get_reward_n_score(obs_dict)
        reward = 0.
        idx_offset = len(q_obs)
        completions = []
        bonus = 0
        # completions_score = []
        for element in tasks_to_complete_traj:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                q_obj_obs[element_idx - idx_offset] -
                goal[element_idx])
            complete = distance < BONUS_THRESH
            # print(distance)
            if complete:
                completions.append(element)
                bonus += (BONUS_THRESH - distance) / BONUS_THRESH
                # completions_score.append(1.0 if 4-len(tasks_to_complete_traj) == TASK_ELEMENTS.index(element) else 0.5)
        # remove tasks when complete
        # [tasks_to_complete_traj.remove(element) for element in completions]
        
        # bonus = float(len(completions))
        # assert bouns < 2
        
        return bonus, completions

if __name__ == "__main__":
    pipeline()
