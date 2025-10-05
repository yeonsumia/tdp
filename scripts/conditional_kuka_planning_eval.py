import os
import numpy as np
import torch
import pdb
import pybullet as p
import os.path as osp
from os.path import join
import json

import gym
import d4rl

from diffusion.denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch import Trainer
from denoising_diffusion_pytorch.datasets.tamp import KukaDataset
from denoising_diffusion_pytorch.mixer import MixerUnet
from denoising_diffusion_pytorch.temporal_attention import TemporalUnet
from denoising_diffusion_pytorch.utils.rendering import KukaRenderer
import diffusion.utils as utils
# import environments

import imageio
from imageio import get_writer
import torch.nn as nn

from diffusion.models.mlp import TimeConditionedMLP
from diffusion.models import Config

from denoising_diffusion_pytorch.utils.pybullet_utils import get_bodies, sample_placement, pairwise_collision, \
    RED, GREEN, BLUE, BLACK, WHITE, BROWN, TAN, GREY, connect, get_movable_joints, set_joint_position, set_pose, add_fixed_constraint, remove_fixed_constraint, set_velocity, get_joint_positions, get_pose, enable_gravity

from gym_stacking.env import StackEnv
from tqdm import tqdm

import copy

from diffusion.tree.tree import TrajAggTree


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


def get_env_state(robot, cubes, attachments):
    joints = get_movable_joints(robot)
    joint_pos = get_joint_positions(robot, joints)

    for cube in cubes:
        pos, rot = get_pose(cube)
        pos, rot = np.array(pos), np.array(rot)

        if cube in attachments:
            attach = np.ones(1)
        else:
            attach = np.zeros(1)

        joint_pos = np.concatenate([joint_pos, pos, rot, attach], axis=0)

    return joint_pos


def execute(samples, env, render_dim=[256, 256], idx=0):
    postprocess_samples = []
    robot = env.robot
    joints = get_movable_joints(robot)
    gains = np.ones(len(joints))

    cubes = env.cubes
    link = 8

    near = 0.001
    far = 4.0
    projectionMatrix = p.computeProjectionMatrixFOV(60., 1.0, near, far)

    location = np.array([0.8, 1.5, 2.4])
    end = np.array([0.0, 0.0, 0.0])
    viewMatrix = p.computeViewMatrix(location, end, [0, 0, 1])

    attachments = set()

    states = [get_env_state(robot, cubes, attachments)]
    rewards = 0
    ims = []

    for sample in samples[1:]:
        p.setJointMotorControlArray(bodyIndex=robot, jointIndices=joints, controlMode=p.POSITION_CONTROL,
                targetPositions=sample[:7], positionGains=gains)

        attachments = set()
        # Add constraints of objects
        for j in range(4):
            contact = sample[14+j*8]

            if contact > 0.5:
                add_fixed_constraint(cubes[j], robot, link)
                attachments.add(cubes[j])
                env.attachments[j] = 1
            else:
                remove_fixed_constraint(cubes[j], robot, link)
                set_velocity(cubes[j], linear=[0, 0, 0], angular=[0, 0, 0, 0])
                env.attachments[j] = 0


        for i in range(10):
            p.stepSimulation()

        states.append(get_env_state(robot, cubes, attachments))

        _, _, im, _, seg = p.getCameraImage(width=render_dim[0], height=render_dim[1], viewMatrix=viewMatrix, projectionMatrix=projectionMatrix)
        im = np.array(im)
        im = im.reshape((render_dim[0], render_dim[1], 4))[:, :, :3]

        state = env.get_state()
        # print(state)
        reward = env.compute_reward()

        rewards = rewards + reward
        ims.append(im)
        # writer.append_data(im)

    attachments = {}
    env.attachments[:] = 0
    env.get_state()
    reward = env.compute_reward()
    rewards = rewards + reward
    state = get_env_state(robot, cubes, attachments)

    # writer.close()

    return state, states, ims, rewards


def eval_episode(model, env, dataset, traj_agg_tree, savepath_i, seed, is_render=True):
    state = env.reset(seed=seed)

    samples_full_list = []
    obs_dim = dataset.obs_dim

    samples = torch.Tensor(state[..., :-4])
    samples = (samples - dataset.mins) / (dataset.maxs - dataset.mins + 1e-8)
    samples = samples[None, None, None].cuda()
    samples = (samples - 0.5) * 2

    conditions = [
           (0, obs_dim, samples),
    ]

    rewards = 0
    frames = []

    counter = 0
    map_tuple = {}
    for i in range(4):
        for j in range(4):
            if i == j:
                continue

            map_tuple[(i, j)] = counter
            counter = counter + 1

    total_samples = []

    for i in range(3):
        stack = env.goal[env.progress+1]
        place = env.goal[env.progress]
        cond_idx = map_tuple[(stack, place)]
        samples = samples_orig = trainer.ema_model.guided_conditional_sample(model, args.batch_size, conditions, cond_idx, stack, place, pg=args.pg, pg_scale=args.pg_scale, guide_step=args.guide_step)

        if args.use_tree or args.use_sub_tree:
            if args.use_sub_tree:
                possible_indices = torch.arange(args.sample_jump_size, args.sample_ub, step=args.sample_jump_size)  # Indices: {args.sample_jump_size, 2 * args.sample_jump_size, ..., h-1}
                sub_plan_index = possible_indices[torch.randperm(len(possible_indices))[:8]]
                sub_plan_index, _ = torch.sort(sub_plan_index)
                sub_plan_index = sub_plan_index.repeat(args.batch_size // 8)
                # print(sub_plan_index.shape, sub_plan_index)
                
                sub_conditions = [[(i, obs_dim, samples[j, i, :]) for i in range(sub_plan_index[j])] for j in range(len(sub_plan_index))]
                
                sub_samples = trainer.ema_model.fast_guided_conditional_sample(model, args.batch_size, sub_conditions, samples, cond_idx, stack, place, args.diffusion_step)
                print(f"sub_samples: {sub_samples.shape}")
                
                state_of_tree = to_np(samples)
                state_of_sub_tree = to_np(sub_samples)
                
                tmp_samples = torch.clamp(samples, -1, 1)
                samples_unscale = (tmp_samples + 1) * 0.5
                unnormed_samples = dataset.unnormalize(samples_unscale)
                stack_xy = unnormed_samples[..., 64:, 7+stack*8:9+stack*8]
                place_xy = unnormed_samples[..., 64:, 7+place*8:9+place*8]
                norm_val = -torch.abs(stack_xy - place_xy).mean(dim=-1).mean(dim=-1)
                main_value = norm_val
                
                tmp_sub_samples = torch.clamp(sub_samples, -1, 1)
                sub_samples_unscale = (tmp_sub_samples + 1) * 0.5
                unnormed_sub_samples = dataset.unnormalize(sub_samples_unscale)
                stack_xy = unnormed_sub_samples[..., 64:, 7+stack*8:9+stack*8]
                place_xy = unnormed_sub_samples[..., 64:, 7+place*8:9+place*8]
                norm_val = -torch.abs(stack_xy - place_xy).mean(dim=-1).mean(dim=-1)
                sub_values = norm_val
                
                best_idx = torch.argmax(main_value)
                best_sub_idx = torch.argmax(sub_values)
                # print(main_value.shape, sub_values.shape)
                print(f"Best idx: {best_idx}, Best value: {main_value[best_idx]}")
                print(f"Best sub idx: {best_sub_idx}, Best sub value: {sub_values[best_sub_idx]}")
                
                is_sub_best = False
                if main_value[best_idx] < sub_values[best_sub_idx]:
                    best_idx = best_sub_idx
                    is_sub_best = True
                    print(f"Use sub traj: {best_idx} ({sub_values[best_idx]})")
                else:
                    print(f"Use origin traj: {best_idx} ({main_value[best_idx]})")

                # Merging + Expanding
                plan_of_tree = []
                traj_agg_tree.integrate_trajectories(state_of_tree, guide=main_value, sub_guide=sub_values, sub_plan_timestep=sub_plan_index, sub_trajectories=state_of_sub_tree, is_sub_best=is_sub_best, best_idx=best_idx)
                next_node = traj_agg_tree.best_traj_leaf
                while not next_node.is_root():
                    plan_of_tree.append(next_node.node_state)
                    next_node = next_node._parent
                plan_of_tree.append(state_of_tree[0,0])
                plan_of_tree = plan_of_tree[::-1]
            else:  
                state_of_tree = to_np(samples)
                planning_horizon = state_of_tree.shape[1]
                plan_of_tree = []

                # Merging + Expanding
                traj_agg_tree.integrate_trajectories(state_of_tree)

                plan_of_tree.append(state_of_tree[0,0])
                for i in range(planning_horizon - 1):
                    # Acting 
                    next_sample, selected_key, _, _ = traj_agg_tree.get_next_state_original()
                    plan_of_tree.append(next_sample)
                    # Pruning 
                    traj_agg_tree.pruning(selected_key)
            print(f"Plan Length: {len(plan_of_tree)}")
            plan_of_tree = np.array(plan_of_tree)[None]
            samples = to_torch(plan_of_tree)
        else:
            tmp_samples = torch.clamp(samples, -1, 1)
            samples_unscale = (tmp_samples + 1) * 0.5
            unnormed_samples = dataset.unnormalize(samples_unscale)
            stack_xy = unnormed_samples[..., 64:, 7+stack*8:9+stack*8]
            place_xy = unnormed_samples[..., 64:, 7+place*8:9+place*8]
            norm_val = -torch.abs(stack_xy - place_xy).mean(dim=-1).mean(dim=-1)
            # best sample
            best_idx = torch.argmax(norm_val)
            print(f"best idx: {best_idx}, value: {norm_val[best_idx]}")
            samples = samples[best_idx][None]

        samples = torch.clamp(samples, -1, 1)
        samples_unscale = (samples + 1) * 0.5
        samples = dataset.unnormalize(samples_unscale)

        samples = to_np(samples.squeeze(0).squeeze(0))

        samples, samples_list, frames_new, reward = execute(samples, env, render_dim=[1024, 1024], idx=i)
        frames.extend(frames_new)

        total_samples.extend(samples_list)

        samples = (samples - dataset.mins) / (dataset.maxs - dataset.mins + 1e-8)
        samples = torch.Tensor(samples[None, None, None]).to(samples_orig.device)
        samples = (samples - 0.5) * 2


        conditions = [
               (0, obs_dim, samples),
        ]

        samples_list.append(samples)

        rewards = rewards + reward
        
        if args.use_tree or args.use_sub_tree:
            del traj_agg_tree
            traj_agg_tree = TrajAggTree(tree_lambda=args.tree_lambda, 
                                        traj_dim=obs_dim,
                                        one_minus_alpha=args.one_minus_alpha,
                                        )

    if is_render:
        frames = np.stack(frames, axis=0)

        if args.use_tree:
            savepath_i = os.path.join(savepath_i, f"tree-planning.mp4")
        else:
            savepath_i = os.path.join(savepath_i, f"diffuser-planning.mp4")

        frames_resized = np.zeros((len(frames), 512, 512, 3))
        for i in range(len(frames)):
            frames_resized[i] = np.resize(frames[i], (512, 512, 3))
        writer = get_writer(savepath_i)

        for img in frames:
            writer.append_data(img)

        writer.close()
        
    return rewards

class PosGuide(nn.Module):
    def __init__(self, cube, cube_other):
        super().__init__()
        self.cube = cube
        self.cube_other = cube_other

    def forward(self, x, t):
        cube_one = x[..., 64:, 7+self.cube*8: 7+self.cube*8]
        cube_two = x[..., 64:, 7+self.cube_other*8:7+self.cube_other*8]

        pred = -100 * torch.pow(cube_one - cube_two, 2).sum(dim=-1)
        return pred



def to_np(x):
    return x.detach().cpu().numpy()

def pad_obs(obs, val=0):
    state = np.concatenate([np.ones(1)*val, obs])
    return state

def set_obs(env, obs):
    state = pad_obs(obs)
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])

#---------------------------------- setup ----------------------------------#
import argparse
parser = argparse.ArgumentParser(description='plan')
parser.add_argument('--use_tree', action='store_true')
parser.add_argument('--tree_lambda', type=float, default=0.98)
parser.add_argument('--use_original_tree', action='store_true')
parser.add_argument('--use_sub_tree',  action='store_true')
parser.add_argument('--one_minus_alpha', type=float, default=0.002)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--pg', action='store_true')
parser.add_argument('--pg_scale',  type=float, default=0.25)
parser.add_argument('--guide_step', type=int, default=1)
parser.add_argument('--sample_ub', type=int, default=52)
parser.add_argument('--sample_jump_size', type=int, default=4)
parser.add_argument('--gpuid', default=0, type=int, help='gpu id')
parser.add_argument('--diffusion_step', type=int, default=100)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

#### dataset
H = 128
dataset = KukaDataset(H)

env_name = "multiple_cube_kuka_temporal_convnew_real2_128"
H = 128
T = 1000
seed = 0

diffusion_path = f'logs/{env_name}/'
diffusion_epoch = 650

dataset = KukaDataset(H)
weighted = 5.0
trial = 0


if args.use_tree:
    savepath = f'logs/{env_name}/plans_conditional_weighted{weighted}_{H}_{T}_bs{args.batch_size}_tree_lambda{args.tree_lambda}_th{args.one_minus_alpha}'
elif args.use_sub_tree:
    if args.pg:
        savepath = f'logs/{env_name}/pg_{args.pg_scale}_plans_conditional_weighted{weighted}_{H}_{T}_bs{args.batch_size}_sub_tree_lambda{args.tree_lambda}_th{args.one_minus_alpha}'
    else:
        savepath = f'logs/{env_name}/plans_conditional_weighted{weighted}_{H}_{T}_bs{args.batch_size}_sub_tree_lambda{args.tree_lambda}_th{args.one_minus_alpha}'
elif args.pg:
    savepath = f'logs/{env_name}/pg_{args.pg_scale}_plans_conditional_weighted{weighted}_{H}_{T}_bs{args.batch_size}_vanilla'
else:
    savepath = f'logs/{env_name}/plans_conditional_weighted{weighted}_{H}_{T}_bs{args.batch_size}_vanilla'
    
utils.mkdir(savepath)

## dimensions
obs_dim = dataset.obs_dim
act_dim = 0

#### model
# model = MixerUnet(
#     dim = 32,
#     image_size = (H, obs_dim),
#     dim_mults = (1, 2, 4, 8),
#     channels = 2,
#     out_dim = 1,
# ).cuda()

model = TemporalUnet(
    horizon = H,
    transition_dim = obs_dim,
    cond_dim = H,
    dim = 128,
    dim_mults = (1, 2, 4, 8),
).cuda()

diffusion = GaussianDiffusion(
    model,
    channels = 2,
    image_size = (H, obs_dim),
    timesteps = T,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

#### load reward and value functions
# reward_model, *_ = utils.load_model(reward_path, reward_epoch)
# value_model, *_ = utils.load_model(value_path, value_epoch)
# value_guide = guides.ValueGuide(reward_model, value_model, discount)
env = StackEnv(conditional=True)

trainer = Trainer(
    diffusion,
    dataset,
    env,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                     # turn on mixed precision training with apex
    results_folder = diffusion_path,
)


print(f'Loading: {diffusion_epoch}')
trainer.load(diffusion_epoch)
render_kwargs = {
    'trackbodyid': 2,
    'distance': 10,
    'lookat': [10, 2, 0.5],
    'elevation': 0
}

# x = dataset[0][0].view(1, 1, H, obs_dim).cuda()
# conditions = [
#        (0, obs_dim, x[:, :, :1]),
# ]
trainer.ema_model.eval()
hidden_dims = [128, 128, 128]


config = Config(
    model_class=TimeConditionedMLP,
    time_dim=128,
    input_dim=obs_dim,
    hidden_dims=hidden_dims,
    output_dim=12,
    savepath="",
)

device = torch.device('cuda')
model = config.make()
model.to(device)


ckpt_path = "./logs/kuka_cube_stack_classifier_new3/value_0.99/state_80.pt"
ckpt = torch.load(ckpt_path)

model.load_state_dict(ckpt)


samples_list = []
frames = []

# models = [PosGuide(1, 3), PosGuide(1, 4), PosGuide(1, 2)]

counter = 0
map_tuple = {}
for i in range(4):
    for j in range(4):
        if i == j:
            continue

        map_tuple[(i, j)] = counter
        counter = counter + 1


# Red = block 0
# Green = block 1
# Blue = block 2
# Yellow block 3


rewards =  []

for i in tqdm(range(100)):
    seed += 1

    if args.use_tree or args.use_sub_tree:
        traj_agg_tree = TrajAggTree(tree_lambda=args.tree_lambda, 
                                        traj_dim=obs_dim,
                                        one_minus_alpha=args.one_minus_alpha,
                                        )
        if args.use_tree:
            print(f"Seed ({seed}), TAT planning")
        if args.use_sub_tree:
            print(f"Seed ({seed}), TDP planning")
    else:
        traj_agg_tree = None        
        print(f"Seed ({seed}), Vanllia planning")

    savepath_i = os.path.join(savepath, str(seed))
    if not os.path.exists(savepath_i): 
        os.mkdir(savepath_i)

    reward = eval_episode(model, env, dataset, traj_agg_tree, savepath_i=savepath_i, seed=seed)
    rewards.append(reward)
    print()
    print("rewards mean: ", np.mean(rewards))
    print("rewards std: ", np.std(rewards) / len(rewards) ** 0.5)

    ## save result as a json file
    json_path = join(savepath_i, 'rollout.json')
    json_data = {'score': reward, 'return': reward}
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)

    if args.use_tree or args.use_sub_tree:
        del traj_agg_tree
