from collections import namedtuple
import numpy as np
import torch
import einops
import pdb

import diffuser.utils as utils

Trajectories = namedtuple('Trajectories', 'actions observations observations_render')

# load gold_location

def min_guide(state, history_length, plan_i, task_name='maze2d-medium'):
    # state: [B * H * D]
    # gold_easy_location = np.load("gold_medium_single_location.npy")
    if task_name == 'maze2d-medium':
        gold_location = np.load("gold_medium_single_location.npy")
    elif task_name == 'maze2d-large':
        gold_location = np.load("gold_single_location.npy")
    elif task_name == 'multi2d-medium':
        gold_location = np.load("gold_medium_location.npy")
    elif task_name == 'multi2d-large':
        gold_location = np.load("gold_location.npy")
    gold_state_location = gold_location[plan_i]
    # gold_state_location = np.array([-0.23648487, 0.54600975]) # grid=[3, 8]
    # final_state_location = np.array([0.93654375, 0.75055706]) # grid=[7, 9]
    # gold_state_location = np.array([-0.23648487, -0.88582143]) # grid=[3, 1]
    # gold_state_location = np.array([0.93654375, -0.2721795]) # grid=[7, 4]
    # gold_state_location = np.array([-0.82299918,  0.13691513]) # grid=[1, 6]
    # gold_dist = -np.min(np.abs(state[:, :, :2] - gold_state_location).mean(axis=2), axis=1)
    # history_gold_dist = -np.min(np.abs(state[:, :history_length, :2] - gold_state_location).mean(axis=2), axis=1)
    # final_dist = -np.abs(state[:, :, :2] - final_state_location).mean(axis=2).mean(axis=1)
    # dist = np.where(history_gold_dist > -0.08, 1, gold_dist)
    dist = -np.min(np.abs(state[:, :, :2] - gold_state_location).mean(axis=2), axis=1)
    # dist: [B]
    return dist
    
def conti_guide(state, history_length, plan_i, task_name='maze2d-medium'):
    # state: [B * H * D]
    # gold_easy_location = np.load("gold_medium_single_location.npy")
    if task_name == 'maze2d-medium':
        gold_location = np.load("gold_medium_single_location.npy")
    elif task_name == 'maze2d-large':
        gold_location = np.load("gold_single_location.npy")
    elif task_name == 'multi2d-medium':
        gold_location = np.load("gold_medium_location.npy")
    elif task_name == 'multi2d-large':
        gold_location = np.load("gold_location.npy")
    gold_state_location = gold_location[plan_i]
    dist = -np.abs(state[:, :, :2] - gold_state_location).mean(axis=2).mean(axis=1)
    # dist: [B]
    return dist

class Policy:
    """
    Vanilla diffuser policy from https://github.com/jannerm/diffuser/blob/maze2d/diffuser/guides/policies.py.
    """

    def __init__(self, diffusion_model, normalizer):
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = normalizer.action_dim

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def __call__(self, conditions, debug=False, batch_size=1, guide=False, guide_step=1, plan_i=-1, pg=False):
        conditions = self._format_conditions(conditions, batch_size)

        ## batchify and move to tensor [ batch_size x observation_dim ]
        # observation_np = observation_np[None].repeat(batch_size, axis=0)
        # observation = utils.to_torch(observation_np, device=self.device)

        ## run reverse diffusion process
        sample = self.diffusion_model(conditions, guide=guide, guide_step=guide_step, pg=pg)
        sample = utils.to_np(sample)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        # actions = np.tanh(actions)

        ## extract first action
        action = actions[0, 0]

        # if debug:
        normed_observations = sample[:, :, self.action_dim:]
        values = min_guide(normed_observations, history_length=0, plan_i=plan_i)
        # get max-value trajectory
        print(f"values: {values.shape}")
        best_idx = np.argmax(values)
        normed_observations = normed_observations[best_idx:best_idx+1]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        # if deltas.shape[-1] < observation.shape[-1]:
        #     qvel_dim = observation.shape[-1] - deltas.shape[-1]
        #     padding = np.zeros([*deltas.shape[:-1], qvel_dim])
        #     deltas = np.concatenate([deltas, padding], axis=-1)

        # ## [ batch_size x horizon x observation_dim ]
        # next_observations = observation_np + deltas.cumsum(axis=1)
        # ## [ batch_size x (horizon + 1) x observation_dim ]
        # observations = np.concatenate([observation_np[:,None], next_observations], axis=1)
                
        trajectories = Trajectories(actions, observations, observations)
        return action, trajectories
        # else:
        #     return action


class TATPolicy(Policy):
    """
    Policy for TAT
    """

    def __init__(self, diffusion_model, normalizer, use_tree, use_original_tree):
        self.use_tree = use_tree
        self.use_original_tree = use_original_tree
        self.tree = None
        self.history = []
        self.unnorm_clue_state = None
        self.clue_state = None
        self.clue_state_depth = None
        self.replan_freq = 50
        super().__init__(diffusion_model, normalizer)


    def __call__(self, conditions, debug=False, batch_size=1, plan_i=-1, pg=False, task_name='maze2d-medium'):
        if self.use_tree:
            return self.tat_call(conditions, debug, batch_size, plan_i=plan_i, pg=pg, task_name=task_name)
        elif self.use_original_tree:
            return self.original_tat_call(conditions, debug, batch_size, guide=True, plan_i=plan_i)
        else:
            guide_step = 4
            return super().__call__(conditions, debug, batch_size, guide=True, guide_step=guide_step, plan_i=plan_i, pg=pg)


    def original_tat_call(self, conditions, debug=False, batch_size=1, guide=False, plan_i=-1):
        conditions = self._format_conditions(conditions, batch_size)

        # Sample plans via vanilla diffuser.
        sample = self.diffusion_model(conditions, guide=guide, plan_i=plan_i)
        sample = utils.to_np(sample)

        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        state_of_sample = sample[:, :, self.action_dim:] # only take the observation
        planning_horizon = state_of_sample.shape[1]

        normed_observations = sample[:, :, self.action_dim:]
        observations_render = self.normalizer.unnormalize(normed_observations, 'observations')
        plan_of_tree = []

        # Merging + Expanding
        self.tree.integrate_trajectories(state_of_sample)

        # Get a plan via open-loop planning
        plan_of_tree.append(state_of_sample[0,0])
        for i in range(planning_horizon - 1):
            # Acting 
            next_sample, selected_key, _, _ = self.tree.get_original_next_state()
            plan_of_tree.append(next_sample)

            # Pruning 
            self.tree.pruning(selected_key)

        plan_of_tree = np.array(plan_of_tree)[None]
        observations = self.normalizer.unnormalize(plan_of_tree, 'observations')

        trajectories = Trajectories(None, observations, observations_render)
        return None, trajectories
    
    
    def tat_call(self, conditions, debug=False, batch_size=1, plan_i=-1, pg=False, task_name='maze2d-medium'):
        print(f"current state: {conditions[0]}")
        print(f"clue state: {self.unnorm_clue_state}, depth: {self.clue_state_depth}")
        conditions = self._format_conditions(conditions, batch_size)
        # self.history.append(conditions[0].unsqueeze(1))
        
        # constrain clue state
        if self.unnorm_clue_state is not None and self.clue_state_depth < max(conditions.keys()) and self.clue_state_depth > 20:
            clue_condition = utils.to_torch(self.clue_state, dtype=torch.float32, device='cuda:0')
            conditions[self.clue_state_depth-self.replan_freq] = einops.repeat(clue_condition, 'd -> repeat d', repeat=batch_size)
        print(f"conditions keys: {conditions.keys()}")
        ############################################################
        # Sample plans via vanilla diffuser.
        original_sample = self.diffusion_model(conditions, pg=pg, task_name=task_name)
        sample = utils.to_np(original_sample)

        actions = sample[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        state_of_sample = sample[:, :, self.action_dim:] # only take the observation
        planning_horizon = state_of_sample.shape[1]

        ############################################################
        # Sample sub-plans via vanilla diffuser
        # time-sampling intermediate states
        # Generate possible indices: {a, 2a, ..., <= h}
        a = 15 # jumpy size
        possible_indices = torch.arange(a, planning_horizon, step=a)  # Indices: {a, 2a, ..., h-1}
        original_state_of_sample = original_sample[:, :, self.action_dim:]
        sub_plan_index = possible_indices[torch.randperm(len(possible_indices))[:4]]
        sub_plan_index, _ = torch.sort(sub_plan_index)
        # sub_plan_index = torch.tensor([ 60, 210, 310])
        # sub_plan_index = torch.tensor([100, 160, 260, 310])
        print(f"sub_plan_index: {sub_plan_index}")
        # sub_plan_state_of_sample_list = []
        # Randomly select one index for each batch
            
        # sub_plan_indices = possible_indices[torch.randint(0, len(possible_indices), (batch_size,))]  # Shape: (b,)
            
        # Gather selected rows for each batch
        # conditions[0] = torch.cat([original_state_of_sample[:, sub_plan_index[i], :] for i in range(3)], dim=0)
        conditions[0] = torch.cat([original_state_of_sample[(batch_size // 4)*i:(batch_size // 4)*(i+1), sub_plan_index[i], :] for i in range(4)], dim=0)
        # conditions[planning_horizon-1] = conditions[planning_horizon-1][0].unsqueeze(0).repeat(3 * batch_size, 1)
        # print(f"cond[0]: {conditions[0].shape}, cond[-1]: {conditions[planning_horizon-1].shape}")
            
        sub_plan_sample = self.diffusion_model(conditions, guide=True, task_name=task_name)
        sub_plan_sample = utils.to_np(sub_plan_sample)

        # sub_plan_actions = sub_plan_sample[:, :, :self.action_dim]
        # sub_plan_actions = self.normalizer.unnormalize(sub_plan_actions, 'actions')

        sub_plan_state_of_sample_list = sub_plan_sample[:, :, self.action_dim:] # only take the observation


        ############################################################
        # Unnormalize observations
        normed_observations = sample[:, :, self.action_dim:]
        # normed_sub_plan_observations = sub_plan_sample[:, :, self.action_dim:]
        observations_render = self.normalizer.unnormalize(normed_observations, 'observations')
        plan_of_tree = []

        ############################################################
        # Calculate guidance of sampled trajectories
        # Constrain the historical states when calculating guidance
        history_length = len(self.history)
        histories = np.concatenate(self.history, axis=0)
        histories = self.normalizer.normalize(histories, 'observations')
        history_values = min_guide(histories[None, :, :], history_length, plan_i, task_name=task_name)
        # print(f"history value: {history_values}, history length: {history_length}")
        histories = np.repeat(histories[None, :, :], batch_size, axis=0)
        
        print(histories.shape)
        concat_normed_observations = np.concatenate([
                histories,
                normed_observations[:, 1:, :]
            ], axis=1)
        # print(concat_normed_observations.shape)
        values = min_guide(concat_normed_observations, history_length, plan_i, task_name=task_name)
        
        # TODO
        concat_normed_sub_plan_observations = None
        concat_normed_sub_plan_observations_list = []
        for i in range(4):
            concat_normed_sub_plan_observations = np.concatenate([
                        histories[:batch_size//4],
                        normed_observations[batch_size//4*i:batch_size//4*(i+1), 1:sub_plan_index[i], :],
                        sub_plan_state_of_sample_list[batch_size//4*i:batch_size//4*(i+1)]
                    ], axis=1)
            # print(concat_normed_sub_plan_observations.shape)
            concat_normed_sub_plan_observations_list.append(concat_normed_sub_plan_observations)
        # print(concat_normed_sub_plan_observations.shape)
        sub_values = [min_guide(c, history_length, plan_i, task_name=task_name) for c in concat_normed_sub_plan_observations_list]
        ############################################################
        
        # Merging + Expanding
        # print(sub_values)
        # concat_normed_sub_plan_observations_list = np.concatenate(concat_normed_sub_plan_observations_list, axis=0)
        print(f"concat_normed_sub_plan_observations_list[0]: {concat_normed_sub_plan_observations_list[0].shape}")
        # sub_plan_state_of_sample_list = np.concatenate(sub_plan_state_of_sample_list, axis=0)
        print(f"sub_plan_state_of_sample_list: {sub_plan_state_of_sample_list.shape}")
        self.tree.integrate_trajectories(state_of_sample, guide=values, sub_guide=[v for v_l in sub_values for v in v_l], sub_plan_timestep=sub_plan_index, sub_trajectories=sub_plan_state_of_sample_list)
        
        # update clue node
        clue_node, selected_key, visit_time, max_depth = self.tree.get_clue_state(self.tree._root)
        # get clue policy
        clue_policy = []
        clue_policy.append(state_of_sample[0, 0])
        clue_policy.extend(self.tree.get_clue_policy(clue_node))
        # if len(clue_policy) < self.replan_freq + 1:
        #     for _ in range(self.replan_freq + 1 - len(clue_policy)):
        #         clue_policy.append(clue_policy[-1])
        
        self.clue_state = clue_node.node_state
        self.unnorm_clue_state = self.normalizer.unnormalize(self.clue_state, 'observations')
        self.clue_state_depth = max_depth
        print(f"clue state: {self.unnorm_clue_state}, value: {clue_node.get_future_value()}, max_depth: {max_depth}")

        # Get a plan via open-loop planning
        # plan_of_tree.append(state_of_sample[0,0])
        # for i in range(planning_horizon - 1):
        #     # Acting 
        #     next_sample, selected_key, _, _, num_child = self.tree.get_next_state()
        #     plan_of_tree.append(next_sample)
            
        #     # obs = self.normalizer.unnormalize(next_sample, 'observations')
        #     # print(f"obs: {obs} (depth: {max_depth}, child: {num_child})")
        #     # for k, n in self.tree._root._children.items():
        #     #     obs = self.normalizer.unnormalize(n.node_state, 'observations')
        #     #     print(f"child: {obs}, value: {n.get_value()}")

        #     # Pruning 
        #     self.tree.pruning(selected_key)

        # # Pruning 
        # self.tree.pruning(selected_key)
        clue_policy = np.array(clue_policy)[None]
        # plan_of_tree = np.array(plan_of_tree)[None]
        # print(clue_policy.shape, plan_of_tree.shape)
        # print(clue_policy[0, 0, :], plan_of_tree[0, 0, :])
        # observations = self.normalizer.unnormalize(concat_normed_sub_plan_observations_list[0][0:1], 'observations')
        observations = self.normalizer.unnormalize(clue_policy, 'observations')
        # observations = self.normalizer.unnormalize(plan_of_tree, 'observations')
        trajectories = Trajectories(None, observations, observations_render)
        return None, trajectories


    def reset_tree(self, traj_agg_tree):
        if self.tree is not None:
            del self.tree
        self.tree = traj_agg_tree