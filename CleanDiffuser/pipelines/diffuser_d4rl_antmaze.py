import os

# import d4rl
import gym
import hydra
import numpy as np
import torch
import imageio, cv2
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.d4rl_antmaze_dataset import D4RLAntmazeDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters
from utils import set_seed
import os
import mujoco_py

from D4RL.d4rl.locomotion.maze_env import MazeEnv
from D4RL.d4rl.locomotion.ant import AntEnv
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjRenderContextOffscreen

import sys
sys.path.append('..')
from diffuser.tree.tree import TrajAggTree

RESET = R = 'r'  # Reset position.
GOAL = G = 'g'

HARDEST_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, 0, 0, 0, 1, G, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, G, 0, 1, 0, 0, G, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, G, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 1, G, 0, G, 1, 0, G, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

DTYPE = torch.float
DEVICE = 'cuda:0'

def to_np(x):
	if torch.is_tensor(x):
		x = x.detach().cpu().numpy()
	return x

def to_torch(x, dtype=None, device=None):
	dtype = dtype or DTYPE
	device = device or DEVICE
	if type(x) is dict:
		return {k: to_torch(v, dtype, device) for k, v in x.items()}
	elif torch.is_tensor(x):
		return x.to(device).type(dtype)
		# import pdb; pdb.set_trace()
	return torch.tensor(x, dtype=dtype, device=device)


gym.envs.registration.register(
    id='antmaze-large-diverse-ours-v2',
    entry_point='D4RL.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=2000,
    kwargs={
        'maze_map': HARDEST_MAZE,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5',
        'non_zero_reset':True, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)


@hydra.main(config_path="../configs/diffuser/antmaze", config_name="antmaze", version_base=None)
def pipeline(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    save_path = save_path.replace('-ours', '')
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    
    # Mapping from grid coordinates to world coordinates
    cell_size = 1.0  # approximate size of maze grid cell
    maze_origin = np.array([-4.0, 0.0, 0.1])  # z slightly above ground

    def cell_to_world(x, y):
        return maze_origin + np.array([x * cell_size, y * cell_size, 0.0])

    # Predefine marker positions and colors
    flag_cells = [(1, 6), (5, 6), (5, 10), (1, 10)]
    colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 1]]
    
    dataset = D4RLAntmazeDataset(
        env.get_dataset(), horizon=args.task.horizon, discount=args.discount,
        noreaching_penalty=args.noreaching_penalty,)
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
    fix_sub_mask = torch.zeros((args.num_candidates * args.num_envs, args.task.horizon, obs_dim + act_dim))
    fix_mask[0, :obs_dim] = 1.
    # masking for sub-tree generation
    if args.use_sub_tree:
        sub_plan_indices = torch.randint(   
            low=1, high=args.task.horizon, size=(args.num_candidates * args.num_envs, )
        )
        for i in range(args.num_candidates * args.num_envs):
            fix_sub_mask[i, :sub_plan_indices[i], :obs_dim] = 1.
    # print(f"fix_mask: {fix_mask.shape}")
    loss_weight = torch.ones((args.task.horizon, obs_dim + act_dim))
    loss_weight[0, obs_dim:] = args.action_loss_weight

    # --------------- Diffusion Model --------------------
    agent = DiscreteDiffusionSDE(
        nn_diffusion, None,
        fix_mask=fix_mask, fix_sub_mask=fix_sub_mask, loss_weight=loss_weight, classifier=classifier, ema_rate=args.ema_rate,
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
        # env = gym.make(args.task.env_name)
        # env_eval = gym.vector.make(args.task.env_name, args.num_envs)
        normalizer = dataset.get_normalizer()
        episode_found_goals = []
        episode_rewards = []
        episode_timesteps = []
        
        def norm_xy(xy):
            return torch.tensor(
                (xy - normalizer.mean[:2]) / normalizer.std[:2], device=args.device, dtype=torch.float32
            )
            
            
        goal_grid_lst = [(1, 6), (5, 6), (5, 10), (1, 10)] 
        norm_1st_goal = norm_xy(env.grid_to_xy(goal_grid_lst[0]))
        norm_2nd_goal = norm_xy(env.grid_to_xy(goal_grid_lst[1]))
        norm_3rd_goal = norm_xy(env.grid_to_xy(goal_grid_lst[2]))
        # norm_4th_goal = norm_xy(env.grid_to_xy((3, 8)))
        norm_5th_goal = norm_xy(env.grid_to_xy(goal_grid_lst[3]))
        
        print(f"1st goal: {goal_grid_lst[0]}")
        print(f"2nd goal: {goal_grid_lst[1]}")
        print(f"3rd goal: {goal_grid_lst[2]}")
        # print(f"4th goal: {(3, 8)}")
        print(f"5th goal: {goal_grid_lst[3]}")
        print(f"norm_1st_goal: {norm_1st_goal}")
        print(f"norm_2nd_goal: {norm_2nd_goal}")
        print(f"norm_3rd_goal: {norm_3rd_goal}")
        # print(f"norm_4th_goal: {norm_4th_goal}")
        print(f"norm_5th_goal: {norm_5th_goal}")
        
        
        def get_visited_goal(xy):
            if torch.norm(xy - norm_1st_goal) <= 0.3:
                return 1, goal_grid_lst[0]
            elif torch.norm(xy - norm_2nd_goal) <= 0.3:
                return 2, goal_grid_lst[1]
            elif torch.norm(xy - norm_3rd_goal) <= 0.3:
                return 3, goal_grid_lst[2]
            # elif torch.norm(xy - norm_4th_goal) <= 0.3:
            #     return 4
            elif torch.norm(xy - norm_5th_goal) <= 0.3:
                return 5, goal_grid_lst[3]
            else:
                return None, None
            
        def num_list_match(lst1):
            count_score = 0
            for i in range(len(lst1) - 1):
                for j in range(i + 1, len(lst1)):
                    if lst1[i] < lst1[j]:
                        count_score += 1
            return count_score
        
        prior = torch.zeros((args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)
        sub_prior = torch.zeros((args.num_candidates * args.num_envs, args.task.horizon, obs_dim + act_dim), device=args.device)
        for i in range(args.num_episodes):

            if args.use_tree:
                traj_agg_tree = TrajAggTree(tree_lambda=0.98, 
                                            traj_dim=obs_dim+act_dim,
                                            one_minus_alpha=0,
                                            )
            obs, ep_reward, cum_done, t = env.reset(), 0., 0., 0
            visited_goals = []
            print(f"======================= Episode {i} =======================")
            norm_init = (env.get_xy() - normalizer.mean[:2]) / normalizer.std[:2]
            print(f"init: {env.xy_to_grid(env.get_xy())}")
            norm_goal = norm_xy(env.get_target())
            print(f"(not used) goal: {env.xy_to_grid(env.get_target())}")
             
            print("===========================================================")
            while not np.all(cum_done) and t < 2000 + 1:
                obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                # sample trajectories
                prior[:, 0, :obs_dim] = obs.unsqueeze(0)
                # calculate score
                # if t % 20 == 0:
                #     print(f"xy: {obs[:2]}")
                new_visited_goal, curr_grid = get_visited_goal(obs[:2])
                if new_visited_goal is not None and new_visited_goal not in visited_goals:
                    visited_goals.append(new_visited_goal)
                    print(f"visit {new_visited_goal} / reset to curr grid: {curr_grid}")
                    # reset for new task
                    obs = env.reset_pos(curr_grid)
                    # normalize obs
                    obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                    # sample trajectories
                    prior[:, 0, :obs_dim] = obs.unsqueeze(0)
                    
                    if args.use_sub_tree:
                        # initialize tree
                        pass
                    if len(visited_goals) == 4:
                        break
                    
                
                # prior[0, 0, :obs_dim] = obs
            
                # print(args.use_sub_tree)
                if args.use_sub_tree:
                    # parent branch generation
                    traj, log = agent.sample(
                        prior.repeat(args.num_candidates, 1, 1),
                        solver=args.solver,
                        n_samples=args.num_candidates * args.num_envs,
                        sample_steps=args.sampling_steps,
                        use_ema=args.use_ema, w_cg=args.task.w_cg, temperature=args.temperature, # also need to use CG !!!
                        # information for seq_match_path task
                        norm_goal_lst=[norm_1st_goal, norm_2nd_goal, norm_3rd_goal, norm_5th_goal],
                        visited_goals=visited_goals,
                        pg=args.pg,
                    )
                    
                    traj_sub, log_sub = agent.sample(
                        prior.repeat(args.num_candidates, 1, 1),
                        solver=args.solver,
                        n_samples=args.num_candidates * args.num_envs,
                        sample_steps=args.fast_sampling_steps, # TODO: fast planning?
                        use_ema=args.use_ema, w_cg=args.task.w_cg, temperature=args.temperature,
                        warm_start_reference=traj, 
                        # information for seq_match_path task
                        norm_goal_lst=[norm_1st_goal, norm_2nd_goal, norm_3rd_goal, norm_5th_goal],
                        visited_goals=visited_goals,
                        # sub_tree_expansion=False,
                    )
                    # print(f"traj: {traj.shape}, log: {log['log_p'].shape}")
                    # print(f"traj_sub: {traj_sub.shape}, log_sub: {log_sub['log_p'].shape}")
                    
                    
                    # select the best plan
                    traj = traj.view(args.num_candidates, args.num_envs, args.task.horizon, -1)
                    logp = log["log_p"].view(args.num_candidates, args.num_envs, -1).sum(-1)
                    traj_sub = traj_sub.view(args.num_candidates, args.num_envs, args.task.horizon, -1)
                    sub_logp = log_sub["log_p"].view(args.num_candidates, args.num_envs, -1).sum(-1)
                    
                    
                    merged_traj = torch.cat([traj, traj_sub], dim=0)
                    merged_logp = torch.cat([logp, sub_logp], dim=0)
                    logp = merged_logp.view(args.num_candidates * 2, args.num_envs)
                    idx = logp.argmax(0)
                    act = merged_traj[idx, torch.arange(args.num_envs), 0, obs_dim:]
                    act = act.clip(-1., 1.).detach().cpu().numpy().squeeze(0)
                elif args.pg:
                    # conditional PG sampling
                    traj, log = agent.sample(
                        prior.repeat(args.num_candidates, 1, 1),
                        solver=args.solver,
                        n_samples=args.num_candidates * args.num_envs,
                        sample_steps=args.sampling_steps,
                        use_ema=args.use_ema, w_cg=args.task.w_cg, temperature=args.temperature,
                        # information for seq_match_path task
                        norm_goal_lst=[norm_1st_goal, norm_2nd_goal, norm_3rd_goal, norm_5th_goal],
                        visited_goals=visited_goals,
                        pg=True,
                    )   
                elif args.use_tree:
                     # conditional sampling
                    traj, log = agent.sample(
                        prior.repeat(args.num_candidates, 1, 1),
                        solver=args.solver,
                        n_samples=args.num_candidates * args.num_envs,
                        sample_steps=args.sampling_steps,
                        use_ema=args.use_ema, w_cg=args.task.w_cg, temperature=args.temperature,
                        # information for seq_match_path task
                        norm_goal_lst=[norm_1st_goal, norm_2nd_goal, norm_3rd_goal, norm_5th_goal],
                        visited_goals=visited_goals,
                    )
                    # build a TAT
                    state_of_tree = to_np(traj)
                    # print(state_of_tree.shape)
                    traj_agg_tree.integrate_trajectories(state_of_tree)
                    
                    # Merging + Expanding
                    plan_of_tree = []
                    plan_of_tree.append(state_of_tree[0,0])
                    for i in range(args.task.horizon - 1):
                        # Acting 
                        next_sample, selected_key, _, _ = traj_agg_tree.get_next_state_original()
                        plan_of_tree.append(next_sample)
                        # Pruning 
                        traj_agg_tree.pruning(selected_key)
                    # print(f"Plan Length: {len(plan_of_tree)}")
                    plan_of_tree = np.array(plan_of_tree)[None]
                    traj = to_torch(plan_of_tree)
                    
                elif args.use_mcss:
                    # conditional sampling
                    traj, log = agent.sample(
                        prior.repeat(args.num_candidates, 1, 1),
                        solver=args.solver,
                        n_samples=args.num_candidates * args.num_envs,
                        sample_steps=args.sampling_steps,
                        use_ema=args.use_ema, w_cg=args.task.w_cg, temperature=args.temperature,
                        # information for seq_match_path task
                        norm_goal_lst=[norm_1st_goal, norm_2nd_goal, norm_3rd_goal, norm_5th_goal],
                        visited_goals=visited_goals,
                    )
                else:
                    # use classifier guidance
                    traj, log = agent.sample(
                        prior.repeat(args.num_candidates, 1, 1),
                        solver=args.solver,
                        n_samples=args.num_candidates * args.num_envs,
                        sample_steps=args.sampling_steps,
                        use_ema=args.use_ema, w_cg=args.task.w_cg, temperature=args.temperature,
                        # information for seq_match_path task
                        norm_goal_lst=[norm_1st_goal, norm_2nd_goal, norm_3rd_goal, norm_5th_goal],
                        visited_goals=visited_goals,
                    )

                if not args.use_sub_tree and not args.use_tree:
                    # select the best plan
                    logp = log["log_p"].view(args.num_candidates, args.num_envs, -1).sum(-1)
                    idx = logp.argmax(0)
                    act = traj.view(args.num_candidates, args.num_envs, args.task.horizon, -1)[
                        idx, torch.arange(args.num_envs), 0, obs_dim:]
                    act = act.clip(-1., 1.).detach().cpu().numpy().squeeze(0)
                
                if args.use_tree:
                    act = traj[:, 0, obs_dim:]
                    act = act.clip(-1., 1.).detach().cpu().numpy().squeeze(0)

                # step
                # print(act.shape)
                obs, rew, done, info = env.step(act)
                if len(visited_goals) == 4:
                    done = np.ones_like(done, dtype=bool)

                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                # print(done)
                ep_reward += rew
                if t % 200 == 0:
                    print(f'[t={t}] xy: {env.xy_to_grid(obs[:2])}, ')
                    print(f'[t={t}] visited_goals: {visited_goals}, ', flush=True)
                    if not args.use_tree:
                        print(f'[t={t}] logp: {logp[idx, torch.arange(args.num_envs)][0]}', flush=True)

            # clip the reward to [0, 1] since the max cumulative reward is 1
            # episode_rewards.append(np.clip(ep_reward, 0., 1.))
            print(f"========================================================")
            print(f"final xy: {env.xy_to_grid(obs[:2])}")
            ep_rew = num_list_match(visited_goals)
            print(f"episode reward: {ep_rew} / total timesteps: {t}")
            episode_rewards.append(ep_rew)
            episode_timesteps.append(t)
            episode_found_goals.append(len(visited_goals))
            print(f"Mean: {np.mean(episode_rewards, -1)}, Std: {np.std(episode_rewards, -1)}")
            print(f"Found goals: {np.mean(episode_found_goals)}")
            print(f"Total timesteps: {sum(episode_timesteps)} / {len(episode_timesteps) * 2000}")
            print("=========================================================")

        print(f"======================= Evaluation =======================")
        print(f"episode_rewards: {episode_rewards}")
        print(f"episode_timesteps: {episode_timesteps}")
        print(f"episode_found_goals: {episode_found_goals}")
        episode_rewards = np.array(episode_rewards)
        print(f"Mean: {np.mean(episode_rewards, -1)}, Std: {np.std(episode_rewards, -1)}")
        print(f"Total timesteps: {sum(episode_timesteps)} / {len(episode_timesteps) * 2000}")

    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    pipeline()
