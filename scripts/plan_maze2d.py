import json
import numpy as np
from os.path import join
import pdb
import os
import torch

from diffuser.guides.policies import TATPolicy
import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.tree.tree import TrajAggTree
from diffuser.utils.training import cycle
from diffuser.utils.arrays import batch_to_device, to_np, to_device, apply_dict


class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#
args = Parser().parse_args('plan')
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

# logger = utils.Logger(args)
num_eval = 20
seed = 0
replan_freq = 50

#--------------------------------- multi2d-gold-picking -------------------------------#
gold_easy_task = [np.array([1, 6]), # 1: [1, 6]
                  np.array([6, 6]),
                  np.array([7, 6]),
                  np.array([7, 4]), # 4
                  np.array([1, 10]), # 5
                  np.array([3, 6]), # 6
                  np.array([2, 6]), # 7
                  np.array([3, 3]),
                  np.array([3, 1]),
                  np.array([5, 10]), # 10
                  np.array([2, 10]), # 11
                  np.array([6, 2]),
                  np.array([3, 4]),
                  np.array([1, 9]),
                  np.array([3, 6]), # 15
                  np.array([7, 8]), # 16
                  np.array([1, 1]),
                  np.array([1, 3]),
                  np.array([1, 1]),
                  np.array([3, 5]),        
]
#--------------------------------- maze2d-gold-picking -------------------------------#
gold_easy_single_task = [np.array([1, 6]), # 1: [1, 6]
                        np.array([1, 9]),
                        np.array([4, 10]), # 3
                        np.array([1, 7]), # 4
                        np.array([5, 9]), # 5
                        np.array([2, 6]), # 6
                        np.array([5, 9]), # 7
                        np.array([3, 3]), # 8
                        np.array([4, 1]),
                        np.array([6, 6]), # 10
                        np.array([3, 10]), # 11
                        np.array([2, 4]),
                        np.array([5, 10]), # 13
                        np.array([5, 9]),
                        np.array([7, 10]), # 15
                        np.array([6, 4]), # 16
                        np.array([1, 6]),
                        np.array([6, 6]),
                        np.array([7, 6]),
                        np.array([7, 4]),        
]
#--------------------------------- multi2d-medium-gold-picking -------------------------------#
gold_easy_medium_task = [np.array([2, 4]), # 1
                        np.array([4, 1]),
                        np.array([4, 4]), # 3
                        np.array([4, 5]), # 4
                        np.array([4, 2]), # 5
                        np.array([6, 6]), # 6
                        np.array([3, 2]), # 7
                        np.array([3, 4]), # 8
                        np.array([1, 2]),
                        np.array([2, 4]), # 10
                        np.array([2, 6]), # 11
                        np.array([4, 5]), # 12
                        np.array([5, 3]), # 13
                        np.array([4, 4]),
                        np.array([2, 6]), # 15
                        np.array([1, 5]), # 16
                        np.array([4, 6]),
                        np.array([1, 2]),
                        np.array([2, 2]),
                        np.array([1, 6]),        
]
#--------------------------------- maze2d-medium-gold-picking -------------------------------#
gold_easy_medium_single_task = [np.array([3, 3]), # 1
                                np.array([3, 4]),
                                np.array([2, 2]), # 3
                                np.array([5, 4]), # 4
                                np.array([3, 4]), # 5
                                np.array([2, 4]), # 6
                                np.array([4, 4]), # 7
                                np.array([3, 4]), # 8
                                np.array([3, 4]),
                                np.array([2, 5]), # 10
                                np.array([6, 3]), # 11
                                np.array([6, 2]), # 12
                                np.array([5, 4]), # 13
                                np.array([5, 4]),
                                np.array([3, 3]), # 15
                                np.array([4, 4]), # 16
                                np.array([6, 3]),
                                np.array([4, 2]),
                                np.array([3, 2]),
                                np.array([2, 5]),        
]

#---------------------------------- loading ----------------------------------#
# task clarification
if args.task_name == 'maze2d-medium':
    target_task = gold_easy_medium_single_task
    args.dataset = 'maze2d-medium-v1'
elif args.task_name == 'multi2d-medium':
    target_task = gold_easy_medium_task
    args.dataset = 'maze2d-medium-v1'
elif args.task_name == 'maze2d-large':
    target_task = gold_easy_single_task
    args.dataset = 'maze2d-large-v1'
elif args.task_name == 'multi2d-large':
    target_task = gold_easy_task
    args.dataset = 'maze2d-large-v1'
    
diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

#--------------------------------- policy -------------------------------#
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


policy = TATPolicy(diffusion, dataset.normalizer, use_tree = args.use_tree, use_original_tree=args.use_original_tree)

gold_easy_location = [dataset.normalizer.normalize([*l, 0, 0], 'observations')[:2] for l in gold_easy_task]
gold_easy_single_location = [dataset.normalizer.normalize([*l, 0, 0], 'observations')[:2] for l in gold_easy_single_task]
gold_medium_location = [dataset.normalizer.normalize([*l, 0, 0], 'observations')[:2] for l in gold_easy_medium_task]
gold_medium_single_location = [dataset.normalizer.normalize([*l, 0, 0], 'observations')[:2] for l in gold_easy_medium_single_task]

# save numpy
np.save("gold_location.npy", gold_easy_location)
np.save("gold_single_location.npy", gold_easy_single_location)
np.save("gold_medium_location.npy", gold_medium_location)
np.save("gold_medium_single_location.npy", gold_medium_single_location)


#---------------------------------- main loop ----------------------------------#

gold_reward = []
for task_i in range(num_eval):
    seed += 1
    env = datasets.load_environment(args.dataset)
    env.seed(seed)

    if args.use_tree or args.use_original_tree:
        traj_agg_tree = TrajAggTree(tree_lambda=args.tree_lambda, 
                                     traj_dim=observation_dim,
                                     one_minus_alpha=args.one_minus_alpha,
                                    )
        policy.reset_tree(traj_agg_tree)
        if args.use_original_tree:
            print(f"Seed ({seed}), TAT planning")
        if args.use_tree:
            print(f"Seed ({seed}), TDP planning")
    else:
        print(f"Seed ({seed}), Vanllia planning")

    savepath_i = os.path.join(args.savepath, str(seed))
    if not os.path.exists(savepath_i): 
        os.mkdir(savepath_i)

    observation = env.reset()

    if args.multi_task:
        print('Resetting target')
        env.set_target()

    ## set conditioning xy position to be the goal
    target = env._target
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
        # int(diffusion.horizon // 3) - 1: np.array([2, 1, 0, 0]),
    }
    
    print(f"reset observation: {observation}")
    # env.set_state(cond[0][:2], cond[0][2:])
    # observation = env._get_obs()
    # print(f"current observation: {observation}")

    print(f"reset target: {target}")
    print(f"planning batch size: {args.batch_size}")
    print(f"max episode steps: {env.max_episode_steps}")
    # env.set_target(target_location=cond[diffusion.horizon - 1][:2])
    # print(f"current target: {env._target}")
    # observations for rendering
    rollout = [observation.copy()]

    total_reward = 0
    found_gold = False
    gold_threshold = 0.3
    gold_dist = gold_threshold # threshold
    for t in range(env.max_episode_steps):

        state = env.state_vector().copy()

        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        
        # [closd-loop] history store
        if t == 0:
            policy.history.append(np.expand_dims(observation, axis=0))
        
        if t == 0: # open loop
            # should check cond[0] & observation from env.reset()
            cond[0] = observation
            cond_draw = cond.copy()

            _, samples = policy(cond, batch_size=args.batch_size, plan_i=task_i, pg=args.pg, task_name=args.task_name)
            tree_observations_render = samples.observations[0].copy()
            # print(f"tree_observations_render: {tree_observations_render.shape}")
            # actions = samples.actions[0]
            sequence = samples.observations[0]
        # pdb.set_trace()

        # ####
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1] # open loop
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            # pdb.set_trace()

        ## can use actions or define a simple controller based on state predictions
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        # pdb.set_trace()
        ####

        # else:
        #     actions = actions[1:]
        #     if len(actions) > 1:
        #         action = actions[0]
        #     else:
        #         # action = np.zeros(2)
        #         action = -state[2:]
        #         pdb.set_trace()



        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        
        # print(
        #     f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        #     f'{action}'
        # )

        if 'maze2d' in args.dataset:
            xy = next_observation[:2]
            goal = env.unwrapped._target
            # print(np.linalg.norm(xy - gold_easy_task[task_i]))
            if np.linalg.norm(xy - target_task[task_i]) < gold_dist:
                print(f"[Task {task_i}] Found gold!")
                # total_reward += 500
                gold_dist = np.linalg.norm(xy - target_task[task_i])
                found_gold = True
            # print(
            #     f'maze | pos: {xy} | goal: {goal}'
            # )

        score = env.get_normalized_score(total_reward)
        ## update rollout observations
        rollout.append(next_observation.copy())

        # logger.log(score=score, step=t)

        if t % args.vis_freq == 0 or terminal:
            fullpath = join(savepath_i, f'{t}.png')

            if t == 0: 
                # print first 10 plans sampled from vanilla diffuser
                for k in range(min(10, len(samples.observations_render))):
                    renderer.composite(join(savepath_i, f'diffuser_plan{k}.png'), samples.observations_render[k][None], start=cond_draw[0], end=cond_draw[diffusion.horizon - 1], ncol=1)
                    # print(samples.observations_render[k][None], cond_draw[0], cond_draw[diffusion.horizon - 1])
                    # renderer.composite(fullpath, samples.observations_render[:4], ncol=1)
                if args.use_tree or args.use_original_tree:
                    renderer.composite(join(savepath_i, f'tree_plan.png'), np.array(tree_observations_render)[None], start=cond_draw[0], end=cond_draw[diffusion.horizon - 1], ncol=1)


            # renderer.render_plan(join(savepath_i, f'{t}_plan.mp4'), samples.actions, samples.observations, state)
            ## save rollout thus far
            renderer.composite(join(savepath_i, 'rollout.png'), np.array(rollout)[None], start=cond_draw[0], end=cond_draw[diffusion.horizon - 1], ncol=1)

            # renderer.render_rollout(join(savepath_i, f'rollout.mp4'), rollout, fps=80)

            # logger.video(rollout=join(savepath_i, f'rollout.mp4'), plan=join(savepath_i, f'{t}_plan.mp4'), step=t)

        if terminal:
            break

        observation = next_observation

    # logger.finish(t, env.max_episode_steps, score=score, value=0)

    ## save result as a json file
    json_path = join(savepath_i, 'rollout.json')
    json_data = {'seed': seed, 'score': score, 'step': t, 'return': total_reward, 'term': terminal,
        'epoch_diffusion': diffusion_experiment.epoch}
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
    print(f"[Task {task_i}] Reward: {total_reward} Gold: {found_gold}")
    if total_reward > 0 and found_gold:
        gold_reward.append((gold_threshold-gold_dist) * 10 / 3)
    else:
        gold_reward.append(0)
    print(f"Gold reward: {gold_reward[-1]}")
    

    env.close()
    del env
    if args.use_tree or args.use_original_tree:
        del traj_agg_tree
        policy.history = []
        
print(gold_reward)
print(f"Total {num_eval} tasks gold reward: {np.mean(gold_reward)} std: {np.std(gold_reward)}")
