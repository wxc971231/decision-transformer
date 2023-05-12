import sys
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../..'))
sys.path.append(base_path)

from PredNet.code.utils_Pred import CfgNode as CN
import torch.nn as nn

class DRR(nn.Module):
    '''
    Best delta RTG Regressor
    '''
    @staticmethod
    def get_default_config():
        C = CN()
        C.hidden1_dim = None
        C.hidden2_dim = None
        C.state_dim = None
        C.dropout = 0.1
        return C
    
    def __init__(self, config):
        super().__init__()
        assert config.hidden1_dim is not None
        assert config.hidden2_dim is not None
        assert config.state_dim is not None
        
        self.config = config
        self.hidden1 = nn.Linear(2*config.state_dim, config.hidden1_dim)
        self.hidden2 = nn.Linear(config.hidden1_dim, config.hidden2_dim)
        self.out_dr = nn.Linear(config.hidden2_dim, 1)
        
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()

    def forward(self, states_pair):
        '''
        states_pair: (batch_size, 2*state_dim)
        '''
        out = self.hidden1(states_pair)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.hidden2(out)
        out = self.relu(out)
        out = self.dropout(out)

        pred_dr = self.out_dr(out)

        return pred_dr

class BAR(nn.Module):
    '''
    Best action classifier
    '''
    @staticmethod
    def get_default_config():
        C = CN()
        C.hidden1_dim = None
        C.hidden2_dim = None
        C.state_dim = None
        C.action_dim = None
        C.dropout = 0.1
        return C
    
    def __init__(self, config):
        super().__init__()
        assert config.hidden1_dim is not None
        assert config.hidden2_dim is not None
        assert config.state_dim is not None
        assert config.action_dim is not None
        
        self.config = config
        self.hidden1 = nn.Linear(2*config.state_dim, config.hidden1_dim)
        self.hidden2 = nn.Linear(config.hidden1_dim, config.hidden2_dim)
        self.out_a = nn.Linear(config.hidden2_dim, config.action_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()

    def forward(self, states_pair):
        '''
        states_pair: (batch_size, 2*state_dim)
        '''
        out = self.hidden1(states_pair)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.hidden2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        pred_a = self.out_a(out)

        return pred_a
    
class PredNet(nn.Module):
    @staticmethod
    def get_default_config():
        C = CN()
        C.DRR = DRR.get_default_config()
        C.BAR = BAR.get_default_config()
        return C
    
    def __init__(self, config):
        super().__init__()
        self.DRR = DRR(config.DRR)
        self.BAR = BAR(config.BAR)

    def forward(self, states_pair):
        pred_a = self.BAR(states_pair)
        pred_dr = self.DRR(states_pair)

        return pred_dr, pred_a
