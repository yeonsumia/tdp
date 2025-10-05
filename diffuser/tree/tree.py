import numpy as np
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def cosine_similarity(x, y):
    similarity = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return similarity

def get_weight(step, tree_lambda):
    return np.power(tree_lambda, step-1) 

def get_value_weight(step, value_lambda, length, history_length):
    return np.power(value_lambda, length-step-history_length)
    # for negative values, use this
    # return 1

class TreeNode(object):
    '''
    A node in the TAT.
    Each node keeps track of its own state, visiting states, weights, and step (for debug)
    '''

    def __init__(self, parent, state, tree_lambda=0.99):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._tree_lambda = tree_lambda
        self._states = [state['state']]
        self._steps = [state['step']]
        self._weights = [state['weight']]
        self._v_weights = [state['v_weight']]
        self._values = [state['value']]
        self.node_state = np.average(np.array(self._states), axis=0, weights=np.array(self._weights)+1e-10)

    
    @property
    def _num_children(self):
        return len(self._children)

    def expand(self, state):
        '''
        Expand tree by creating new children.
        '''
        self._children[self._num_children] = TreeNode(self, state, self._tree_lambda)
        return self._children[self._num_children - 1]


    def is_children(self, state, dis_threshold):
        '''
        Find most suitable node to transition.
        '''
        min_distance = 9999
        min_distance_key = None
        for key in self._children.keys():
            node_state = self._children[key].node_state
            distance = cosine_similarity(state, node_state)
            distance = 1 - distance
            if distance < min_distance:
                min_distance = distance
                min_distance_key = key
        if min_distance < dis_threshold:
            return True, min_distance_key
        else:
            return False, None
            
            
    def update_children(self, state, key):
        '''
        Update the statistics of this node.
        '''
        self._children[key]._states.append(state['state'])
        self._children[key]._steps.append(state['step'])
        self._children[key]._weights.append(state['weight'])
        self._children[key]._v_weights.append(state['v_weight'])
        self._children[key]._values.append(state['value'])
        self._children[key].update_node_state()


    def update_node_state(self,):
        '''
        Update the state of this node.
        '''
        self.node_state = np.average(np.array(self._states), axis=0, weights=np.array(self._weights))


    def get_value(self):
        '''
        Return the total weights for this node.
        '''
        # return np.sum(np.array(self._weights))
        # should be mean (not summation) 
        # also should consider reliable trajectory via weight summation (to lower stochastic risk)
        if self._values[0] < 0:
            return np.dot(np.array(self._values), np.array(self._v_weights)+1e-10) / len(np.array(self._v_weights))
        else:
            return np.sum(np.array(self._weights))
        
    
    def get_original_value(self):
        '''
        Return the total weights for this node.
        '''
        return np.sum(np.array(self._weights))
            

    def get_future_value(self):
        '''
        Return the discounted score for this node.
        '''
        # return np.dot(np.array(self._values), np.array(self._weights)+1e-10) / len(np.array(self._weights))
        # return np.dot(np.array(self._values), np.array(self._v_weights)+1e-10) / len(np.array(self._v_weights))
        if self._values[0] < 0:
            return np.mean(np.array(self._values))
        else:
            return np.sum(np.array(self._weights))

    def step(self, key):
        '''
        Transition to a specific child node.
        '''
        return self._children[key]


    def is_leaf(self):
        '''
        check if leaf node (i.e. no nodes below this have been expanded).
        '''
        return self._children == {}

    def is_root(self):
        '''
        check if it's root node
        '''
        return self._parent is None


class TrajAggTree(object):
    '''
    An implementation of Trajectory Aggregation Tree (TAT).
    '''
    def __init__(self, tree_lambda, traj_dim, action_dim=None, one_minus_alpha=0.005, start_state=None):
        self._tree_lambda = tree_lambda
        self._distance_threshold = one_minus_alpha # 1-\alpha
        self.traj_dim = traj_dim
        self.action_dim = action_dim
        if start_state is None:
            start_state = np.zeros((traj_dim,))
        state = {'state': start_state, 'step': 0, 'weight': 1, 'value': 1, 'v_weight': 1}
        self._root = TreeNode(None, state, self._tree_lambda)


    def integrate_single_traj(self, traj, length, history_length, value=1, node=None, sub_plan_timestep=None, sub_traj=None, sub_value=None, sub_step=0):
        '''
        Integrate a single trajectory into the tree.
        '''
        if node is None:
            node = self._root

        node_i = []
        # Merging the former sub-trajectory
        for i in range(history_length, length):
            if node.is_leaf():
                break
            is_children, key = node.is_children(traj[i], self._distance_threshold)

            if is_children:
                state = {
                            'state': traj[i], 
                            'step': sub_step + i, 
                            'weight': get_weight(i, self._tree_lambda), 
                            'v_weight': get_value_weight(i, self._tree_lambda, length, history_length), 
                            'value': value
                        }
                node.update_children(state, key=key)
                node = node.step(key)
                if sub_plan_timestep is not None and i+1 in sub_plan_timestep:
                    node_i.append(node)
            else:
                # no suitable nodes for transition
                break
        
        # Expanding the latter sub-trajectory
        if i < length - 1:
            for j in range(i, length):
                state = {
                            'state': traj[j], 
                            'step': sub_step + j, 
                            'weight': get_weight(j, self._tree_lambda), 
                            'v_weight': get_value_weight(j, self._tree_lambda, length, history_length), 
                            'value': value
                        }
                node = node.expand(state)
                if sub_plan_timestep is not None and j+1 in sub_plan_timestep:
                    node_i.append(node)
        if sub_plan_timestep is not None:
            # print(f"found nodes: {node_i}")
            for z in range(len(sub_plan_timestep)):
                # print(f"sub_traj[z]: {sub_traj[z].shape}, sub_plan_timestep[z]: {sub_plan_timestep[z].shape}")
                self.integrate_single_traj(sub_traj[z], length, history_length, value=sub_value[z], node=node_i[z], sub_step=sub_plan_timestep[z])

        

    def integrate_trajectories(self, trajectories, guide=None, history_length=1, sub_guide=None, sub_plan_timestep=None, sub_trajectories=None):
        '''
        Integrate a batch of new trajectories sampled from diffusion planners.
        history_length: trajectories contain historical states (e.g., one history state in Diffuser). We will not integrate the historical part.
        '''
        assert len(trajectories.shape) == 3 and trajectories.shape[-1] == self.traj_dim

        batch_size, length = trajectories.shape[0], trajectories.shape[1]        
        for i in range(batch_size):
            # trajectories[i]: (H, D)
            # guide[i]: (1,)
            # sub_plan_timesteps[i]: (1,)
            # sub_trajectories[i]: (H, D)
            if guide is not None:
                sub_plan_timestep_lst = [sub_plan_timestep[i // (batch_size // 4)]]
                sub_val_list = [sub_guide[i]]
                sub_traj_list = [sub_trajectories[i]]
                self.integrate_single_traj(trajectories[i], length, history_length, value=guide[i], sub_value=sub_val_list, sub_plan_timestep=sub_plan_timestep_lst, sub_traj=sub_traj_list)
            else:
                self.integrate_single_traj(trajectories[i], length, history_length)



    def get_next_state(self,):
        '''
        Acting: select the most impactful node, which has highest weight among the child nodes.
        '''
        selected_key, node = max(self._root._children.items(), key=lambda node: node[1].get_value())
        visit_time = len(node._states)
        max_depth = np.array(node._steps).max()
        # if len(self._root._children.items()) > 1:
            # print(f"fork: {node.node_state} (depth: {max_depth})")
        return node.node_state, selected_key, visit_time, max_depth, len(self._root._children.items())
    
    def get_original_next_state(self,):
        '''
        Acting: select the most impactful node, which has highest weight among the child nodes.
        '''
        selected_key, node = max(self._root._children.items(), key=lambda node: node[1].get_original_value())
        visit_time = len(node._states)
        max_depth = np.array(node._steps).max()
        # if len(self._root._children.items()) > 1:
            # print(f"fork: {node.node_state} (depth: {max_depth})")
        return node.node_state, selected_key, visit_time, max_depth
    
    
    def get_clue_state(self, root):
        '''
        Predicting: find the most impactful node in the future via BFS
        '''
        if root.is_leaf():
            return None, None, 0, -1
        
        selected_key, node = max(root._children.items(), key=lambda _node: _node[1].get_future_value())
        max_value = node.get_future_value()
        for _, c_n in root._children.items():
            n, k, _, d = self.get_clue_state(c_n)
            if d == -1:
                continue
            v = n.get_future_value()
            if max_value < v:
                selected_key, node, max_value = k, n, v
        
        visit_time = len(node._states)
        max_depth = np.array(node._steps).max()
        
        return node, selected_key, visit_time, max_depth
    
    def get_clue_policy(self, clue_node):
        n = clue_node
        policy = []
        # next states for padding
        i = 0
        next_node = n
        while len(next_node._children) > 0:
            _, next_node = max(next_node._children.items(), key=lambda z: z[1].get_value())
            policy.append(next_node.node_state)
            # i += 1
        policy.reverse()
        while n != None:
            policy.append(n.node_state)
            n = n._parent
        policy = policy[:-1]
        policy.reverse()
        print(f"len policy: {len(policy)}")
        return policy
        
        

    def pruning(self, selected_key):
        '''
        Pruning: prune the tree, keeping in sync with the environment.
        '''
        self._root = self._root._children[selected_key]
        self._root._parent = None


    def forward_state(self, trajectories, action_dim=None, first_action=None):
        if action_dim is not None and action_dim != 0:
            _actions = trajectories[:, :, :self.action_dim]
            if first_action is None:
                first_action = np.zeros_like(_actions)[:, 0, :][:, None, :]
            _actions = _actions[:,:-1,:] # discard the last action
            _actions = np.concatenate([first_action, _actions], axis=1)
            _observations = trajectories[:, :, self.action_dim:]
            tree_trajectories = np.concatenate([_actions, _observations], axis=-1)
        else:
            tree_trajectories = trajectories
        return tree_trajectories
    
    def reverse_state(self, tree_trajectories, action_dim=None, last_action=None):
        if action_dim is not None and action_dim != 0:
            _actions = tree_trajectories[:, :, :self.action_dim]
            if last_action is None:
                # pad with the current last action
                last_action = _actions[:, -1, :].copy()
                last_action = last_action[:, None, :]
            _actions = _actions[:, 1:, :] # discard the first action
            _actions = np.concatenate([_actions, last_action], axis=1)
            _observations = tree_trajectories[:, :, self.action_dim:]
            trajectories = np.concatenate([_actions, _observations], axis=-1)
        else:
            trajectories = tree_trajectories
        return trajectories

    def __str__(self):
        return "TrajAggTree"