import os
import os.path as osp
import numpy as np
import torch
import pdb
import pybullet as p
import argparse
import math

from diffusion.denoising_diffusion_pytorch_pnp import GaussianDiffusion  # TODO
from denoising_diffusion_pytorch import Trainer
from denoising_diffusion_pytorch.datasets.tamp import KukaDataset
from denoising_diffusion_pytorch.mixer import MixerUnet
from denoising_diffusion_pytorch.temporal_attention import TemporalUnet
from denoising_diffusion_pytorch.utils.rendering import KukaRenderer
import diffusion.utils as utils
# import environments
from imageio import get_writer
import torch.nn as nn

from diffusion.models.mlp import TimeConditionedMLP
from diffusion.models import Config
from diffusion.tree.tree import TrajAggTree

from gym_stacking.utils import get_bodies, sample_placement, pairwise_collision, \
    RED, GREEN, BLUE, BLACK, WHITE, BROWN, TAN, GREY, connect, get_movable_joints, set_joint_position, set_pose, add_fixed_constraint, remove_fixed_constraint, set_velocity, get_joint_positions, get_pose, enable_gravity

from gym_stacking.pick_env import PickandPutEnv, get_env_state
from tqdm import tqdm


DTYPE = torch.float
DEVICE = 'cuda:0'

def to_torch(x, dtype=None, device=None):
	dtype = dtype or DTYPE
	device = device or DEVICE
	if type(x) is dict:
		return {k: to_torch(v, dtype, device) for k, v in x.items()}
	elif torch.is_tensor(x):
		return x.to(device).type(dtype)
		# import pdb; pdb.set_trace()
	return torch.tensor(x, dtype=dtype, device=device)

def execute(samples, env, idx=0):
    # postprocess_samples = []

    states = [get_env_state(env.robot, env.cubes, env.attachments)]
    rewards = 0
    ims = []

    dists = []
    for ind, sample in enumerate(samples[1:]):
        joint_pos = sample[:7]
        contact = [sample[14+j*8] for j in range(4)]
        action = np.concatenate([joint_pos, contact], dtype=np.float32)

        state, reward, done, _ = env.step(action)
        if env.save_render:
            im = env.render()
            ims.append(im)
            # writer.append_data(im)

        states.append(get_env_state(env.robot, env.cubes, env.attachments))

        if ind < samples.shape[0] - 2:
            actual_state = states[-1]
            pred_state = samples[ind+2]
            vec1 = np.concatenate([actual_state[:7], [actual_state[14+j*8] for j in range(4)]], dtype=np.float32)
            vec2 = np.concatenate([pred_state[:7], [pred_state[14+j*8] for j in range(4)]], dtype=np.float32)
            dist = np.linalg.norm(vec1 - vec2)
            dists.append(dist)

        rewards = rewards + reward

    env.attachments[:] = 0
    env.get_state()  # env.get_state is also set_state
    reward = env.compute_reward()
    rewards = rewards + reward
    state = get_env_state(env.robot, env.cubes, env.attachments)

    # writer.close()

    if np.max(dists) > 1.02:
        print("BAD?")
        rewards = 0.   # TODO: delete bad trajectory in this way

    return state, states, ims, rewards

total_distance = []
def eval_episode(guide, env, dataset, traj_agg_tree, idx=0, args=None):
    state = env.reset()

    # samples_full_list = []
    obs_dim = dataset.obs_dim

    samples = torch.Tensor(state[..., :-4])
    samples = (samples - dataset.mins) / (dataset.maxs - dataset.mins + 1e-8)
    samples = samples[None, None, None].cuda()
    samples = (samples - 0.5) * 2  # [0,1] -> [-1,1]

    conditions = [
           (0, obs_dim, samples),
    ]

    rewards = 0
    frames = []

    total_samples = []
    

    for i in range(4):
        stack = env.goal[env.progress]
        place = env.put_place[stack]
        cond_idx = 0
        # samples = samples_orig = trainer.ema_model.conditional_sample(args.batch_size, conditions)
        samples = samples_orig = trainer.ema_model.guided_conditional_sample(guide, args.batch_size, conditions, cond_idx, stack, place[:2], pg=args.pg, pg_scale=args.pg_scale, guide_step=args.guide_step)
        
        put_place = torch.from_numpy(place[:2]).reshape((1,1,2)).to(device)
        # place_xy = put_place.repeat(1,64,1)
        place_xy = np.repeat(place[:2].reshape((1,1,2)), 64, axis=1)
        # print(samples[-1, -2, 7+stack*8:9+stack*8], place[:2])
        tmp_samples = torch.clamp(samples, -1, 1)
        samples_unscale = (tmp_samples + 1) * 0.5
        unnormed_samples = dataset.unnormalize(samples_unscale)
        unnormed_samples = to_np(unnormed_samples)
        values = -np.abs(unnormed_samples[..., 64:, 7+stack*8:9+stack*8] - place_xy).mean(axis=2).mean(axis=1)
        
        # best value idx
        best_idx = np.argmax(values)
        print(f"Best idx: {best_idx}, Best value: {values[best_idx]}")
        
        if args.use_tree or args.use_sub_tree or args.use_seq_tree:
            state_of_tree = to_np(samples)
            planning_horizon = state_of_tree.shape[1]
            plan_of_tree = []
            
            if args.use_sub_tree:
                possible_indices = torch.arange(args.sample_jump_size, args.sample_ub, step=args.sample_jump_size)  # Indices: {a, 2a, ..., h-1}
                sub_plan_index = possible_indices[torch.randperm(len(possible_indices))[:8]]
                sub_plan_index, _ = torch.sort(sub_plan_index)
                sub_plan_index = sub_plan_index.repeat(args.batch_size // 8)
                # print(sub_plan_index.shape, sub_plan_index)
                
                sub_conditions = [[(i, obs_dim, samples[j, i, :]) for i in range(sub_plan_index[j])] for j in range(len(sub_plan_index))]
                
                sub_samples = trainer.ema_model.fast_guided_conditional_sample(guide, args.batch_size, sub_conditions, samples, cond_idx, stack, place[:2], args.diffusion_step)
                print(f"sub_samples: {sub_samples.shape}")
                
                state_of_sub_tree = to_np(sub_samples)
                
                # values for sub-samples
                sub_samples = torch.clamp(sub_samples, -1, 1)
                sub_samples_unscale = (sub_samples + 1) * 0.5
                sub_samples = dataset.unnormalize(sub_samples_unscale)
                unnormalized_sub_samples = to_np(sub_samples)
                sub_values = -np.abs(unnormalized_sub_samples[..., 64:, 7+stack*8:9+stack*8] - place_xy).mean(axis=2).mean(axis=1) # [B]
                
                
                # sub_values = -torch.abs(sub_samples[..., 64:, 7+stack*8:9+stack*8] - place_xy).mean(dim=-1).mean(dim=-1)
                merged_values = values + sub_values
                best_merged_idx = np.argmax(merged_values)
                best_sub_idx = np.argmax(sub_values)
                
                print(f"Best sub idx: {best_sub_idx}, Best sub value: {sub_values[best_sub_idx]}")
                best_idx = best_merged_idx
                is_sub_best = False
                if values[best_merged_idx] < sub_values[best_merged_idx]:
                    
                    is_sub_best = True
                    print(f"Use sub traj: {best_idx} ({sub_values[best_idx]})")
                else:
                    print(f"Use origin traj: {best_idx} ({values[best_idx]})")

                # Merging + Expanding
                traj_agg_tree.integrate_trajectories(state_of_tree, guide=values, sub_guide=sub_values, sub_plan_timestep=sub_plan_index, sub_trajectories=state_of_sub_tree, is_sub_best=is_sub_best, best_idx=best_idx)
                next_node = traj_agg_tree.best_traj_leaf
                while not next_node.is_root():
                    plan_of_tree.append(next_node.node_state)
                    next_node = next_node._parent
                plan_of_tree.append(state_of_tree[0,0])
                plan_of_tree = plan_of_tree[::-1]
            elif args.use_tree:
                # traj_agg_tree.integrate_trajectories(state_of_tree, values=values)
            # else:
                traj_agg_tree.integrate_trajectories(state_of_tree)
                plan_of_tree.append(state_of_tree[0, 0])
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
            # Monte Carlo Sampling (deploy best sample)
            print(samples.shape)
            samples = samples[best_idx][None]
        # executing_samples = samples.repeat(args.batch_size, 1, 1)
        samples = torch.clamp(samples, -1, 1)
        samples_unscale = (samples + 1) * 0.5
        samples = dataset.unnormalize(samples_unscale)
        print(f"unnorm: {samples[-1, -2, 7+stack*8:9+stack*8]}, {place[:2]}")
        samples = to_np(samples.squeeze(0).squeeze(0))
        
        # print(f"sample shape: {samples.shape}") # [128, 39]
        samples, samples_list, frames_new, reward = execute(samples, env, idx=i)
            
        frames.extend(frames_new)

        # if args.do_generate and reward > 0.5:
        #     np.save(os.path.join(args.gen_dir, "cond_sample_{}.npy".format(idx*4+i)), np.array(samples_list))

        total_samples.extend(samples_list)

        samples = (samples - dataset.mins) / (dataset.maxs - dataset.mins + 1e-8)
        samples = torch.Tensor(samples[None, None, None]).to(samples_orig.device)
        samples = (samples - 0.5) * 2

        conditions = [
               (0, obs_dim, samples),
        ]

        samples_list.append(samples)

        rewards = rewards + reward

        # print("reward: %.4f   " % reward)

        env.progress = env.progress + 1
        
        if args.use_tree or args.use_sub_tree or args.use_seq_tree:
            del traj_agg_tree
            traj_agg_tree = TrajAggTree(tree_lambda=args.tree_lambda, 
                                        traj_dim=obs_dim,
                                        one_minus_alpha=args.one_minus_alpha,
                                        )

    if args is not None:
        save_dir = os.path.join(args.savepath, "cond_samples")
    else:
        save_dir = "cond_samples"

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    if env.save_render:
        if args.use_tree:
            vid_savepath = os.path.join(save_dir, "tree_cond_video_writer{}.mp4".format(idx))
        elif args.use_sub_tree:
            if args.pg:
                vid_savepath = os.path.join(save_dir, "pg_sub_tree_cond_video_writer{}.mp4".format(idx)) 
            else:
                vid_savepath = os.path.join(save_dir, "sub_tree_cond_video_writer{}.mp4".format(idx))    
        else:
            vid_savepath = os.path.join(save_dir, "cond_video_writer{}.mp4".format(idx))
        writer = get_writer(vid_savepath)
        for frame in frames:
            writer.append_data(frame)

    np.save(os.path.join(save_dir, "cond_sample_{}.npy".format(idx)), np.array(total_samples))

    if args.do_generate:
        if rewards > 1.5:
            np.save(os.path.join(args.gen_dir, "cond_sample_{}.npy".format(idx)), np.array(total_samples))

    # writer = get_writer("video_writer.mp4")
    # for frame in frames:
    #     writer.append_data(frame)

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

# def pad_obs(obs, val=0):
#     state = np.concatenate([np.ones(1)*val, obs])
#     return state
#
# def set_obs(env, obs):
#     state = pad_obs(obs)
#     qpos_dim = env.sim.data.qpos.size
#     env.set_state(state[:qpos_dim], state[qpos_dim:])

parser = argparse.ArgumentParser()
parser.add_argument('--suffix', default='0', type=str, help='save dir suffix')
parser.add_argument('--env_name', default="multiple_cube_kuka_temporal_convnew_real2_128", type=str, help='env name')
parser.add_argument('--data_path', default="kuka_dataset", type=str, help='dataset root path')
parser.add_argument('--random_seed', default=128, type=int, help="random seed")
parser.add_argument('--batch_size', default=32, type=int, )

parser.add_argument('--use_tree', action='store_true', help='use tree')
parser.add_argument('--use_sub_tree',  action='store_true')
parser.add_argument('--use_seq_tree',  action='store_true')
parser.add_argument('--freq', type=int, default=10)
parser.add_argument('--tree_lambda', type=float, default=0.98)
parser.add_argument('--one_minus_alpha', type=float, default=0.002)
parser.add_argument('--diffusion_step', type=int, default=100)
parser.add_argument('--pg', action='store_true')
parser.add_argument('--pg_scale', type=float, default=0.5)
parser.add_argument('--guide_step', type=int, default=1)
parser.add_argument('--sample_ub', type=int, default=52)
parser.add_argument('--sample_jump_size', type=int, default=4)
parser.add_argument('--gpuid', default=0, type=int, help='gpu id')

parser.add_argument('--save_render', action='store_true', help='save render')
parser.add_argument('--eval_times', default=100, type=str, help='evaluation times')
parser.add_argument('--do_generate', action='store_true', help='do generate')
parser.add_argument('--diffusion_epoch', default=650, type=int, help="diffusion epoch")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

seed = args.random_seed
np.random.seed(seed)
torch.cuda.manual_seed(seed)

#### dataset
env_name = args.env_name
H = 128
T = 1000
dataset = KukaDataset(H)

diffusion_path = f'logs/{env_name}/'

weighted = 5.0

if args.use_tree:
    savepath = f'logs/{env_name}/tree_plans_weighted{weighted}_{H}_{T}/pick2put/{args.suffix}'
elif args.use_sub_tree:
    if args.pg:
        savepath = f'logs/{env_name}/pg_sub_tree_plans_weighted{weighted}_{H}_{T}_b{args.batch_size}_t{args.diffusion_step}/pick2put/{args.suffix}' # small noise: diffusion_step // 2
    else:
        savepath = f'logs/{env_name}/sub_tree_plans_weighted{weighted}_{H}_{T}_b{args.batch_size}_t{args.diffusion_step}/pick2put/{args.suffix}' # small noise: diffusion_step // 2

else:
    if args.pg:
        savepath = f'logs/{env_name}/pg_plans_weighted{weighted}_{H}_{T}_b{args.batch_size}/pick2put/{args.suffix}'
    else:
        savepath = f'logs/{env_name}/plans_weighted{weighted}_{H}_{T}_b{args.batch_size}/pick2put/{args.suffix}'
utils.mkdir(savepath)

args.savepath = savepath

## dimensions
obs_dim = dataset.obs_dim
# act_dim = 0

print(args)

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
env = PickandPutEnv(conditional=True, save_render=args.save_render)

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


print(f'Loading: {args.diffusion_epoch}')
trainer.load(args.diffusion_epoch)
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
    savepath=savepath,
)

device = torch.device('cuda')
guide = config.make()
guide.to(device)

guide_model_ckpt = "./logs/kuka_cube_stack_classifier_new3/value_0.99/state_80.pt"
ckpt = torch.load(guide_model_ckpt)

guide.load_state_dict(ckpt)

# samples_list = []
# frames = []

# models = [PosGuide(1, 3), PosGuide(1, 4), PosGuide(1, 2)]

#####################################################################
# TODO: Color
# Red = block 0
# Green = block 1
# Blue = block 2
# Yellow block 3
#####################################################################

if args.do_generate:
    gen_dir = os.path.join(args.savepath, "gen_dataset")
    args.gen_dir = gen_dir
    if not osp.exists(gen_dir):
        os.makedirs(gen_dir)

rewards =  []

max_rewards = 0.
max_std = 0.
max_id = 0

for i in tqdm(range(args.eval_times)):
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
    
    reward = eval_episode(guide, env, dataset, traj_agg_tree, idx=i, args=args)
    rewards.append(reward)

    mean_reward = np.mean(rewards)
    print()

    print("rewards mean: ", mean_reward)
    print("rewards std: ", np.std(rewards) / len(rewards) ** 0.5, flush=True)

    if i > 90 and mean_reward >= max_rewards:
        max_rewards = mean_reward
        max_std = np.std(rewards) / len(rewards) ** 0.5
        max_id = i + 1
        
    if args.use_tree or args.use_sub_tree or args.use_seq_tree:
        del traj_agg_tree

print("Max id:", max_id)
print("Max rewards:", max_rewards)
print("Corresponding std:", max_std)