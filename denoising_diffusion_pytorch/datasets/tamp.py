import collections
import numpy as np
import pdb

import torch
from torch.utils.data import Dataset
from glob import glob


def to_tensor(x, dtype=torch.float, device='cpu'):
    return torch.tensor(x, dtype=dtype, device=device)


class TampDataset(Dataset):

    conditions = [
        # ([], 1), ## none
        ([0], 1), ## first
        ([-1], 1), ## last
        ([0,-1], 1), ## first and last
    ]

    def __init__(self, H, max_path_length=1000, max_n_episodes=4000):
        dataset = "/data/vision/billf/scratch/yilundu/pddlstream/output_5/*.npy"
        datasets = sorted(glob(dataset))
        obs_dim = 63

        conditions_k, conditions_p = zip(*self.conditions)

        self.conditions_k = np.array(conditions_k, dtype=np.object)
        self.conditions_p = np.array(conditions_p) / sum(conditions_p)

        qstates = np.zeros((max_n_episodes, max_path_length, obs_dim))
        path_lengths = np.zeros(max_n_episodes, dtype=np.int)

        for i, dataset in enumerate(datasets):
            qstate = np.load(dataset)
            print(qstate.max(), qstate.min())
            # qstate[np.isnan(qstate)] = 0.0
            path_length = len(qstate)

            if path_length > max_path_length:
                qstates[i, :max_path_length] = qstate[:max_path_length]
                path_length = max_path_length
            else:
                qstates[i, :path_length] = qstate
            path_lengths[i] = path_length
        qstates = qstates[:i+1]
        path_lengths = path_lengths[:i+1]

        ## make indices
        print('Making indices')
        indices = []
        for i, path_length in enumerate(path_lengths):
            for start in range(path_length - H + 1):
                end = start + H
                indices.append((i, start, end))
        indices = np.array(indices)

        self.obs_dim = obs_dim
        self.qstates = qstates
        self.path_lengths = path_lengths
        self.indices = indices

        self.normalize()

        print(f'[ TampDataset ] qstates: {qstates.shape}')

    def normalize(self):
        '''
            normalizes to [-1, 1]
        '''
        # dataset = self.env.get_dataset()
        # X = np.concatenate([dataset['infos/qpos'], dataset['infos/qvel']], axis=-1)
        mins = self.mins = self.qstates.min(axis=0).min(axis=0)
        maxs = self.maxs = self.qstates.max(axis=0).max(axis=0)
        ## [ 0, 1 ]
        self.qstates = (self.qstates - mins) / (maxs - mins + 1e-5)
        ## [ -1, 1 ]
        self.qstates = self.qstates * 2 - 1

    def unnormalize(self, x):
        '''
            x : [ 0, 1 ]
        '''
        assert x.max() <= 1 and x.min() >= 0, f'x range: ({x.min():.4f}, {x.max():.4f})'
        mins = to_tensor(self.mins, dtype=x.dtype, device=x.device)
        maxs = to_tensor(self.maxs, dtype=x.dtype, device=x.device)
        return x * (maxs - mins) + mins

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-7):
        path_ind, start, end = self.indices[idx]
        qstates = self.qstates[path_ind, start:end]
        assert qstates.max() <= 1.0 + eps and qstates.min() >= -1.0 - eps, f'qstates range: ({qstates.min():.4f}, {qstates.max():.4f})'

        cond = np.random.choice(self.conditions_k, p=self.conditions_p)
        qstates = to_tensor(qstates[None])
        mask = torch.zeros_like(qstates)
        for t in cond:
            mask[:, t] = 1

        return qstates, mask


class KukaDataset(Dataset):

    conditions = [
        # ([], 1), ## none
        ([0], 1), ## first
        ([-1], 1), ## last
        ([0,-1], 1), ## first and last
    ]

    def __init__(self, H, max_path_length=300, max_n_episodes=15600):
        dataset = "kuka_dataset/*.npy"
        datasets = sorted(glob(dataset))
        obs_dim = 39

        conditions_k, conditions_p = zip(*self.conditions)

        self.conditions_k = np.array(conditions_k, dtype=object)
        self.conditions_p = np.array(conditions_p) / sum(conditions_p)

        # qstates = np.zeros((max_n_episodes, max_path_length, obs_dim))
        # path_lengths = np.zeros(max_n_episodes, dtype=np.int64)

        # for i, dataset in enumerate(datasets):
        #     qstate = np.load(dataset)
        #     qstate = qstate[::2]
        #     print(qstate.max(), qstate.min())
        #     # qstate[np.isnan(qstate)] = 0.0
        #     path_length = len(qstate)

        #     if path_length > max_path_length:
        #         qstates[i, :max_path_length] = qstate[:max_path_length]
        #         path_length = max_path_length
        #     else:
        #         qstates[i, :path_length] = qstate
        #     path_lengths[i] = path_length
        # qstates = qstates[:i+1]
        # path_lengths = path_lengths[:i+1]

        ## make indices
        # print('Making indices')
        # indices = []
        # for i, path_length in enumerate(path_lengths):
        #     for start in range(path_length - H + 1):
        #         end = start + H
        #         indices.append((i, start, end))
        # indices = np.array(indices)

        self.obs_dim = obs_dim
        # self.qstates = qstates
        # self.path_lengths = path_lengths
        # self.indices = indices

        # self.normalize()
        self.mins = np.array([-2.966681755887534, -2.094236466504906, -2.966880414803569, -1.9266493808762153, -2.9650449002282895, -2.0939916650236854, -3.054109216240107, -0.9755592346191406, -0.972710371017456, 0.0, -0.861461044658356, -0.8607871562850816, -0.865970275150423, -0.4999703937574057, 0.0, -0.9700714945793152, -0.9731171727180481, 0.0, -0.8612350930892974, -0.8622241841839282, -0.8658882947885027, -0.4999913992231425, 0.0, -0.9741771221160889, -0.9720529317855835, 0.0, -0.8644780314351191, -0.8656788746682704, -0.8659614701012672, -0.49998125060979864, 0.0, -0.9727120399475098, -0.9735767841339111, 0.0, -0.8629993575900159, -0.8634915969032311, -0.8658516998221809, -0.49998328913699525, 0.0])
        self.maxs = np.array([2.9668503033048537, 2.0940478742921442, 2.9664354273814983, 2.059066062442449, 2.9668407283585245, 2.0932704869289367, 3.0541642018946704, 0.9676653146743774, 0.9706068634986877, 1.3356753587722778, 0.9998920205627997, 0.9999565290677489, 0.9999999999053404, 1.0, 1.0, 0.9707653522491455, 0.9719069004058838, 1.3356900215148926, 0.9997936427392372, 0.9999664965667753, 0.9999999968785491, 1.0, 1.0, 0.975237250328064, 0.9739353656768799, 1.335688591003418, 0.999988227487957, 0.9998442570316121, 0.9999999309781705, 1.0, 1.0, 0.9738906621932983, 0.9678307175636292, 1.3356369733810425, 0.9998533905191259, 0.999966236245177, 0.999999957315429, 1.0, 1.0])
            
        # print(f'[ TampDataset ] qstates: {qstates.shape}')

    def normalize(self):
        '''
            normalizes to [-1, 1]
        '''
        # dataset = self.env.get_dataset()
        # X = np.concatenate([dataset['infos/qpos'], dataset['infos/qvel']], axis=-1)
        mins = self.mins = self.qstates.min(axis=0).min(axis=0)
        maxs = self.maxs = self.qstates.max(axis=0).max(axis=0)
        ## [ 0, 1 ]
        self.qstates = (self.qstates - mins) / (maxs - mins + 1e-5)
        ## [ -1, 1 ]
        self.qstates = self.qstates * 2 - 1
        
        print(list(self.mins), list(self.maxs))

    def unnormalize(self, x):
        '''
            x : [ 0, 1 ]
        '''
        assert x.max() <= 1 and x.min() >= 0, f'x range: ({x.min():.4f}, {x.max():.4f})'
        mins = to_tensor(self.mins, dtype=x.dtype, device=x.device)
        maxs = to_tensor(self.maxs, dtype=x.dtype, device=x.device)
        return x * (maxs - mins) + mins

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-7):
        path_ind, start, end = self.indices[idx]
        qstates = self.qstates[path_ind, start:end]
        assert qstates.max() <= 1.0 + eps and qstates.min() >= -1.0 - eps, f'qstates range: ({qstates.min():.4f}, {qstates.max():.4f})'

        cond = np.random.choice(self.conditions_k, p=self.conditions_p)
        qstates = to_tensor(qstates)
        # mask = qstates[-1]
        # for t in cond:
        #     mask[:, t] = 1
        mask = torch.zeros_like(qstates[..., -1])
        for t in cond:
            mask[t] = 1

        return qstates, mask


class KukaDatasetReward(Dataset):

    conditions = [
        # ([], 1), ## none
        ([0], 1), ## first
        ([-1], 1), ## last
        ([0,-1], 1), ## first and last
    ]

    def __init__(self, H, max_path_length=1000, max_n_episodes=12000):
        dataset = "kuka_dataset/*.npy"
        datasets = sorted(glob(dataset))
        obs_dim = 39

        conditions_k, conditions_p = zip(*self.conditions)

        self.conditions_k = np.array(conditions_k, dtype=np.object)
        self.conditions_p = np.array(conditions_p) / sum(conditions_p)

        qstates = np.zeros((max_n_episodes, max_path_length, obs_dim))
        path_lengths = np.zeros(max_n_episodes, dtype=np.int)

        for i, dataset in enumerate(datasets):
            qstate = np.load(dataset)
            qstate = qstate[::2]
            print(qstate.max(), qstate.min())
            # qstate[np.isnan(qstate)] = 0.0
            path_length = len(qstate)

            if path_length > max_path_length:
                qstates[i, :max_path_length] = qstate[:max_path_length]
                path_length = max_path_length
            else:
                qstates[i, :path_length] = qstate
            path_lengths[i] = path_length
        qstates = qstates[:i+1]
        path_lengths = path_lengths[:i+1]

        ## make indices
        print('Making indices')
        indices = []
        for i, path_length in enumerate(path_lengths):
            for start in range(path_length - H + 1):
                end = start + H
                indices.append((i, start, end))
        indices = np.array(indices)

        self.obs_dim = obs_dim

        positions = []
        for i in range(4):
            pos = qstates[:, :, 7+i*8:10+i*8]
            positions.append(pos)

        labels = []

        for i in range(4):
            for j in range(4):
                if i == j:
                    continue

                pos_i = positions[i]
                pos_j = positions[j]

                pos_stack = np.linalg.norm(pos_i[..., :2] - pos_j[..., :2], axis=-1) < 0.1
                height_stack = pos_i[..., 2] > pos_j[..., 2]

                stack = pos_stack & height_stack
                labels.append(stack)

        self.labels = np.stack(labels, axis=-1)

        self.qstates = qstates
        self.path_lengths = path_lengths
        self.indices = indices

        self.normalize()

        print(f'[ TampDataset ] qstates: {qstates.shape}')

    def normalize(self):
        '''
            normalizes to [-1, 1]
        '''
        # dataset = self.env.get_dataset()
        # X = np.concatenate([dataset['infos/qpos'], dataset['infos/qvel']], axis=-1)
        mins = self.mins = self.qstates.min(axis=0).min(axis=0)
        maxs = self.maxs = self.qstates.max(axis=0).max(axis=0)
        ## [ 0, 1 ]
        self.qstates = (self.qstates - mins) / (maxs - mins + 1e-5)
        ## [ -1, 1 ]
        self.qstates = self.qstates * 2 - 1

    def unnormalize(self, x):
        '''
            x : [ 0, 1 ]
        '''
        assert x.max() <= 1 and x.min() >= 0, f'x range: ({x.min():.4f}, {x.max():.4f})'
        mins = to_tensor(self.mins, dtype=x.dtype, device=x.device)
        maxs = to_tensor(self.maxs, dtype=x.dtype, device=x.device)
        return x * (maxs - mins) + mins

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-7):
        path_ind, start, end = self.indices[idx]
        qstates = self.qstates[path_ind, start:end]
        assert qstates.max() <= 1.0 + eps and qstates.min() >= -1.0 - eps, f'qstates range: ({qstates.min():.4f}, {qstates.max():.4f})'

        target = self.labels[path_ind, start:end]

        return qstates, target
