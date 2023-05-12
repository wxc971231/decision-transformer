import sys
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../..'))
sys.path.append(base_path)

import torch
from torch.utils.data import Dataset
from typing import List

import numpy as np
import json
import pickle
from PredNet.code.utils_Pred import CfgNode as CN

class StatePairDataset(Dataset):
    @staticmethod
    def get_default_config():
        C = CN()
        C.env_name = None
        C.data_type = None
        C.pred_len = None
        C.split_test_set = None
        C.data_split_seed = 1
        return C
    
    def __init__(self, config:CN, trajectories:List, type:str): 
        self.config = config

        self.trajectories = trajectories
        self.state_pairs = []
        self.trans_action = []
        self.delta_rtg = []
        self.distance = []

        assert config.pred_len is not None
        assert config.env_name is not None
        assert config.data_type is not None
        assert config.split_test_set in [True, False]

        data_name = f'{config.env_name}-{config.data_type}'
        data_saved_path = f'{base_path}/gym/PredNet/data/{data_name}_{type}.json'
        if os.path.exists(data_saved_path):
            self._load_data(data_saved_path)
        else:
            for traj in trajectories:
                obss = traj['observations']
                actions = traj['actions']
                rewards = traj['rewards']
                
                for i,s in enumerate(obss):
                    for j,s_ in enumerate(obss[i+1: min(config.pred_len, obss.shape[0])]):
                        if not np.array_equal(s, s_):
                            s_pair = np.concatenate((s,s_),axis=0)
                            self.state_pairs.append(s_pair.tolist())
                            self.trans_action.append(actions[i].tolist())
                            self.distance.append(j+1)
                            self.delta_rtg.append(sum([r for r in rewards[i:i+j+1]]))
            self._save_data(data_saved_path)
    
        self.state_pairs = torch.tensor(self.state_pairs)   # (sample_num, 2*obs_dim)
        self.trans_action = torch.tensor(self.trans_action) # (sample_num, act_dim)
        self.delta_rtg = torch.tensor(self.delta_rtg)       # (sample_num, )
        self.distance = torch.tensor(self.distance)         # (sample_num, )

    def _save_data(self, data_saved_path):
        data = {'state_pairs':self.state_pairs,
                'trans_action': self.trans_action, 
                'delta_rtg': self.delta_rtg,
                'distance': self.distance,}                     
        jsdata = json.dumps(data) 
        with open(data_saved_path, 'w') as f:
            f.write(jsdata) 

    def _load_data(self, data_saved_path):
        with open(data_saved_path, 'r') as f:
            data = json.load(f)
        self.state_pairs = data['state_pairs']
        self.trans_action = data['trans_action']
        self.delta_rtg = data['delta_rtg']
        self.distance = data['distance']

    def __len__(self):
        return len(self.distance)

    def __getitem__(self, idx):
        return self.state_pairs[idx], self.trans_action[idx], self.distance[idx], self.delta_rtg[idx]
    
    def get_action_dim(self):
        return int(self.trans_action.shape[1])
    
    def get_state_dim(self):
        assert self.state_pairs.shape[1] % 2 == 0
        return int(self.state_pairs.shape[1] / 2)
    
def create_dataset(config, episodes:List):
    if config.split_test_set:
        idxs = np.arange(len(episodes))
        idx = int(0.8*idxs.shape[0])
        data_split_rng = np.random.RandomState(config.data_split_seed)
        data_split_rng.shuffle(idxs)
        episodes = np.array(episodes)
        episodes_train = episodes[idxs[:idx]]
        episodes_test = episodes[idxs[idx:]]
        train_dataset = StatePairDataset(config, episodes_train, type=f'seed{config.data_split_seed}_train')
        test_dataset = StatePairDataset(config, episodes_test, type=f'seed{config.data_split_seed}_test')
    else:
        train_dataset = StatePairDataset(config, episodes, type='all')
        test_dataset = None

    return train_dataset, test_dataset