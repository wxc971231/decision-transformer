import sys
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../..'))
sys.path.append(base_path)

import time
import math
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from PredNet.code.utils_Pred import CfgNode as CN
from torch.nn import functional as F

class Trainer:
    @staticmethod
    def get_default_config():
        C = CN()
        
        # device to train on
        C.device = 'auto'
        
        # dataloder parameters
        C.num_workers = 0

        # optimizer parameters
        C.max_epoch = None
        C.batch_size = 64
        C.betas = (0.9, 0.95)
        C.learning_rate_DRR = 0.01
        C.learning_rate_BAR = 0.01
        C.temperature_DRR = 0.5
        C.temperature_BAR = 0.5
        return C

    def __init__(self, config, model, train_dataset, test_dataset):
        self.config = config.trainer
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if self.config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = self.config.device
        self.model = self.model.to(self.device)
        self.model.device = self.device
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.epoch_num = 0
        self.epoch_time = 0.0
        self.epoch_dt = 0.0
        self.epoch_losses = []

        # setup the dataloader
        self.train_loader = DataLoader(
            self.train_dataset,
            #sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
        if self.test_dataset is not None:
            self.test_loader = DataLoader(
                self.test_dataset,
                #sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
                shuffle=False,
                pin_memory=True,
                batch_size=len(self.test_dataset),
                num_workers=self.config.num_workers,
            )
        else:
            self.test_loader = None

        '''
        # for pred a acc test    
        self.states_pairs_2D = []
        self.states_pairs_best_a = []
        self.states_pairs_best_dr = []
        self.states_pairs_distance = []
        '''
        
    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def _pred_act_cnt(self, real_bas, pred_a):
        pred_a = torch.max(pred_a, axis=1).indices
        pred_a = pred_a.unsqueeze(-1).expand(-1,real_bas.shape[1])
        pred_a = pred_a + 1
        return real_bas.shape[0], torch.any((real_bas==pred_a), axis=1).sum().item()

    def get_test_loss(self):
        assert self.test_loader is not None
        self.model.eval()
        epoch_losses_a = []
        epoch_losses_dr = []
        #pred_losses_bdr = []
        #pred_act_right_cnt = 0
        #pred_act_cnt = 0
        config = self.config
        T_BAR, T_DRR = config.temperature_BAR, config.temperature_DRR
        with torch.no_grad():    
            for batch in self.test_loader:
                batch = [t.to(self.device) for t in batch]
                #real_bdr = batch[4].type(torch.float).unsqueeze(-1)     # (batch_size, 1)
                #real_bas = batch[5].type(torch.long)                    # (batch_size, vocab_size)
                states_pair = batch[0].type(torch.float)                # (batch_size, 2*state_dim)
                delta_rtg = batch[3].type(torch.float).unsqueeze(-1)    # (batch_size, 1)
                trans_action = batch[1].type(torch.float)               # (batch_size, act_dim)
                distance = batch[2].type(torch.float)                   # (batch_size,)
                distance_factor_DRR = math.exp(1/T_DRR)/torch.exp(distance/T_DRR)
                distance_factor_BAR = math.exp(1/T_BAR)/torch.exp(distance/T_BAR)

                # forward the model
                pred_dr, pred_a = self.model(states_pair)               # pred_dr (batch_size, 1);  pred_a (batch_size, act_dim)
                loss_dr = F.mse_loss(pred_dr, delta_rtg, reduction='none').squeeze()    # (batch_size,)
                loss_a = F.mse_loss(pred_a, trans_action, reduction='none').mean(dim=1)      # (batch_size,)
                loss_dr = torch.mul(loss_dr, distance_factor_DRR).mean()
                loss_a = torch.mul(loss_a, distance_factor_BAR).mean()

                '''
                # check performance
                pred_cnt, right_cnt = self._pred_act_cnt(real_bas, pred_a)
                pred_act_cnt += pred_cnt
                pred_act_right_cnt += right_cnt
                loss_bdr = F.mse_loss(pred_dr, real_bdr, reduction='none').squeeze()        # (batch_size,)
                loss_bdr = torch.mul(loss_bdr, distance_factor_DRR).mean()
                '''

                # loss summary
                epoch_losses_a.append(loss_a.item())
                epoch_losses_dr.append(loss_dr.item())
                #pred_losses_bdr.append(loss_bdr.item())
            
        epoch_loss_a = np.array(epoch_losses_a).mean().item()
        epoch_loss_dr = np.array(epoch_losses_dr).mean().item()
        #pred_loss_bdr = np.array(pred_losses_bdr).mean().item()

        self.model.train()
        return epoch_loss_a, epoch_loss_dr

    '''
    def get_test_acc(self, max_distance, map):
        def _get_best_path_info(start, end):
            dis, ret, best_rodes, best_actions = map.floyd(start, end, start, 
                                                        current_rode=[start,], 
                                                        current_actios=[],
                                                        best_rodes=[],
                                                        best_actions=[])
            best_actions = np.unique(np.array(best_actions)[:,0]).tolist()
            return float(ret), best_rodes, best_actions

        if self.states_pairs_2D == []:
            for s0 in range(map.ncol):
                for s1 in range(map.nrow):
                    for s_0 in range(map.ncol):
                        for s_1 in range(map.nrow):
                            dis = math.sqrt((s_0-s0)*(s_0-s0) + (s_1-s1)*(s_1-s1))
                            if not (s0==s_0 and s1==s_1) and dis < max_distance \
                                and not np.array_equal(np.array([s0, s1]), map._target_location):
                                self.states_pairs_2D.append((s0,s1,s_0,s_1))
                                
                                best_dr,_,best_actions = _get_best_path_info(start=map._to_1D_state(s0,s1), end=map._to_1D_state(s_0,s_1))
                                self.states_pairs_best_a.append(best_actions)
                                self.states_pairs_best_dr.append(best_dr)
                                self.states_pairs_distance.append(dis)
                                
            self.states_pairs_2D = torch.tensor(self.states_pairs_2D, dtype=torch.float).reshape(-1,4).to(self.device)
            self.states_pairs_best_dr = torch.tensor(self.states_pairs_best_dr, dtype=torch.float).unsqueeze(-1).to(self.device)
            self.states_pairs_distance = torch.tensor(self.states_pairs_distance, dtype=torch.float).to(self.device)

        # 用被测模型预测所有 state_pair 间的最佳 dr 和 a
        self.model.eval()
        with torch.no_grad(): 
            pred_dr, pred_as = self.model(self.states_pairs_2D)
        pred_as = torch.max(pred_as, axis=1).indices.unsqueeze(-1)
        pred_as = pred_as + 1

        # 计算预测的最佳 dr 和真实最佳 dr 间的加权 MSE loss
        T_DRR =  self.config.temperature_DRR
        loss_dr = F.mse_loss(pred_dr, self.states_pairs_best_dr, reduction='none').squeeze()    # (batch_size,)
        distance_factor_DRR = math.exp(1/T_DRR)/torch.exp(self.states_pairs_distance/T_DRR)
        loss_dr = torch.mul(loss_dr, distance_factor_DRR).mean().item()

        # 遍历所有 states_pair 计算真实标记，进而计算预测精度
        pred_a_right_cnt = 0
        for i, best_a in enumerate(self.states_pairs_best_a):            
            if pred_as[i].item() in best_a:
                pred_a_right_cnt += 1
    
        self.model.train()
        return loss_dr, pred_a_right_cnt/len(self.states_pairs_best_a)
    '''

    def run(self):
        assert self.config.max_epoch is not None
        model, config = self.model, self.config

        # setup the optimizer
        optimizer_DRR = torch.optim.AdamW(model.DRR.parameters(), lr=config.learning_rate_DRR, betas=config.betas)
        optimizer_BAR = torch.optim.AdamW(model.BAR.parameters(), lr=config.learning_rate_BAR, betas=config.betas)
        self.optimizer = {}
        self.optimizer['DRR'] = optimizer_DRR
        self.optimizer['BAR'] = optimizer_BAR

        self.epoch_num = 0
        self.epoch_time = time.time()
        self.epoch_loss_a = 0.0
        self.epoch_loss_dr = 0.0
        epoch_losses = []
        epoch_losses_a = []
        epoch_losses_dr = []
        #data_iter = iter(train_loader)
        T_BAR, T_DRR = config.temperature_BAR, config.temperature_DRR
        while True:
            model.train()
            for batch in self.train_loader:
                batch = [t.to(self.device) for t in batch]     
                states_pair = batch[0].type(torch.float)                # (batch_size, 2*state_dim)
                delta_rtg = batch[3].type(torch.float).unsqueeze(-1)    # (batch_size, 1)
                trans_action = batch[1].type(torch.float)               # (batch_size, act_dim)
                distance = batch[2].type(torch.float)                   # (batch_size,)
                distance_factor_DRR = math.exp(1/T_DRR)/torch.exp(distance/T_DRR)
                distance_factor_BAR = math.exp(1/T_BAR)/torch.exp(distance/T_BAR)

                # forward the model
                pred_dr, pred_a = self.model(states_pair)               # pred_dr (batch_size, 1);  pred_a (batch_size, act_dim)
                loss_dr = F.mse_loss(pred_dr, delta_rtg, reduction='none').squeeze()        # (batch_size,)
                loss_a = F.mse_loss(pred_a, trans_action, reduction='none').mean(dim=1)     # (batch_size,)
                loss_dr = torch.mul(loss_dr, distance_factor_DRR).mean()
                loss_a = torch.mul(loss_a, distance_factor_BAR).mean()

                # backprop and update the parameters
                model.DRR.zero_grad(set_to_none=True)
                model.BAR.zero_grad(set_to_none=True)
                self.optimizer['DRR'].zero_grad()
                self.optimizer['BAR'].zero_grad()
                loss_a.backward()
                loss_dr.backward()
                self.optimizer['DRR'].step()
                self.optimizer['BAR'].step()

                epoch_losses_a.append(loss_a.item())
                epoch_losses_dr.append(loss_dr.item())

            self.epoch_loss_a = np.array(epoch_losses_a).mean().item()
            self.epoch_loss_dr = np.array(epoch_losses_dr).mean().item()
            self.trigger_callbacks('on_epoch_end')

            self.epoch_num += 1
            epoch_losses.clear()
            tnow = time.time()
            self.epoch_dt = tnow - self.epoch_time
            self.epoch_time = tnow

            # termination conditions
            if self.epoch_num >= config.max_epoch:
                break
