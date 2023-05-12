import sys
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
sys.path.append(base_path)

import torch
import json
import numpy as np
from PredNet.code.model_Pred import PredNet
from PredNet.code.utils_Pred import load_data, set_seed, create_empty_floder, CfgNode as CN
import pickle

def load_config(config_path, env_name, data_type):
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        config = CN(**{'system':CN(**config_dict['system']), 
                        'trainer':CN(**config_dict['trainer']),
                        'data':CN(**config_dict['data']),
                        'model':CN(**{'DRR':CN(**config_dict['model']['DRR']), 
                                    'BAR': CN(**config_dict['model']['BAR'])})})
        assert config.data.env_name == env_name
        assert config.data.data_type == data_type
    return config

def get_spairs_and_dr(config, trajectories):
    # 使用预测模型在原始轨迹中连接捷径
    delta_rtg = []
    state_pairs = []
    traj_index = {}  # {traj_idx:[[pair_idx,], [(i,j),]]}
    pair_idx = 0
    for traj_idx, traj in enumerate(trajectories):
        obss = traj['observations']
        rewards = traj['rewards']
        traj_index[traj_idx] = [[], []]

        for i,s in enumerate(obss):
            for j,s_ in enumerate(obss[i+1: min(config.data.pred_len, obss.shape[0])]):
                if not np.array_equal(s, s_):
                    s_pair = np.concatenate((s,s_),axis=0)
                    state_pairs.append(s_pair)
                    delta_rtg.append(sum([r for r in rewards[i:i+j+1]]))
                    traj_index[traj_idx][0].append(pair_idx)
                    traj_index[traj_idx][1].append((i,i+j+1))
                    pair_idx += 1
                    
    state_pairs = np.concatenate(state_pairs)             
    state_pairs = torch.tensor(state_pairs).reshape(-1, 2*config.model.DRR.state_dim).cuda()
    delta_rtg = torch.tensor(delta_rtg, dtype=torch.float).cuda()     

    return traj_index, state_pairs, delta_rtg

def use_model_to_pred(model, state_pairs):
    '''
    用模型预测给定 state_pair 之间的最佳 action 和 delta_rtg
    '''
    model.eval()
    with torch.no_grad(): 
        pred_drs, pred_as = model(state_pairs)                  
    pred_drs = pred_drs.squeeze()                       # (sum_pair_num, )
    pred_as = pred_as.squeeze()                         # (sum_pair_num, action_dim)

    return pred_drs, pred_as

def get_max_improve(config, model, trajectories):
    '''
    计算每条轨迹中进行一次short_path的最大提升
    '''
    traj_index, state_pairs, delta_rtg = get_spairs_and_dr(config, trajectories)
    pred_drs, _ = use_model_to_pred(model, state_pairs)
    epi_max_improve = [0]*len(trajectories)
    for traj_idx, _ in enumerate(trajectories):
        pair_idx = traj_index[traj_idx][0]
        traj_delta_rtg = delta_rtg[pair_idx]                # (pair_num, )
        traj_pred_drs = pred_drs[pair_idx]                  # (pair_num, )

        rtg_improves = traj_pred_drs - traj_delta_rtg       # (pair_num, )
        rtg_improves_max = torch.max(rtg_improves).item()
        if rtg_improves_max > 0:
            epi_max_improve[traj_idx] = rtg_improves_max

    return epi_max_improve


def bootstrap_short_path(config, model, max_improve, short_path_ratio, episodes, times=0):
    def make_one_short_path():
        '''
        考察 episodes 中所有轨迹，如果使用 model 连接一条捷径得到的提升可以超过 max_improve*short_path_ratio，则连接一次捷径
        '''
        epi_index, state_pairs, delta_rtg = get_spairs_and_dr(config, episodes)
        pred_drs, pred_as = use_model_to_pred(model, state_pairs)

        no_change = True
        for epi_idx, epi in enumerate(episodes):
            obss = epi['observations']
            obss_next = epi['next_observations']
            actions = epi['actions']
            rewards = epi['rewards']
            terminals = epi['terminals']
            epi_len = obss.shape[0]

            pair_idx = epi_index[epi_idx][0]
            epi_delta_rtg = delta_rtg[pair_idx]                 # (pair_num, )
            epi_pred_drs = pred_drs[pair_idx]                   # (pair_num, )
            epi_pred_as = pred_as[pair_idx]                     # (pair_num, )

            rtg_improves = epi_pred_drs - epi_delta_rtg         # (pair_num, )
            rtg_improves_max = torch.max(rtg_improves).item()
            if rtg_improves_max > max_improve[epi_idx]*short_path_ratio:
                no_change = False
                replace_pos = torch.argmax(rtg_improves).item()

                i, j = epi_index[epi_idx][1][replace_pos]
                hat_a = epi_pred_as[replace_pos]
                hat_dr = epi_pred_drs[replace_pos]
                
                assert abs(epi_delta_rtg[replace_pos].item() - rewards[i:j].sum()) < 0.0001
                assert epi_len == obss[:i+1].shape[0] + obss[j:].shape[0] + (j-i-1)

                epi['observations'] = np.concatenate((obss[:i+1], obss[j:]))
                epi['next_observations'] = np.concatenate((obss_next[:i+1], obss_next[j:]))
                epi['terminals'] = np.concatenate((terminals[:i+1], terminals[j:]))
                epi['actions'] = np.concatenate((actions[:i+1], actions[j:]))
                epi['rewards'] = np.concatenate((rewards[:i+1], rewards[j:]))
                epi['actions'][i] = hat_a.cpu().numpy()
                epi['rewards'][i] = hat_dr.item()
                        
        return no_change, episodes
    
    assert (times==0) ^ (short_path_ratio==0)
    if times == 0:
        # bootstrap 地进行提升，直到没有任何捷径连接可以将 RTG 提升超过 max_improve[epi_idx]*short_path_ratio
        done, episodes = make_one_short_path()
        while not done:
            done, episodes = make_one_short_path()
        return episodes
    else:
        for _ in range(times):
            done, episodes = make_one_short_path()
            if done:
                break
        return episodes
'''
def no_overlap_short_path(config, model, max_improve, short_path_ratio, episodes):        
    def solit_by_short_path(episode, improve_threshold):     
        obss = episode['observations']
        obss_next = episode['next_observations']
        actions = episode['actions']
        rewards = episode['rewards']
        terminals = episode['terminals']

        if obss.size == 0:
            return []
        elif obss.shape[0] == 1:
            return []

        epi_index, state_pairs, delta_rtg = get_spairs_and_dr(config, [episode,])
        pred_drs, pred_as = use_model_to_pred(model, state_pairs)
        
        if state_pairs.shape[0] == 1:
            pred_drs = pred_drs.unsqueeze(0)
            
        pair_idx = epi_index[0][0]
        epi_delta_rtg = delta_rtg[pair_idx]                 # (pair_num, )
        epi_pred_drs = pred_drs[pair_idx]                   # (pair_num, )
        
        rtg_improves = epi_pred_drs - epi_delta_rtg         # (pair_num, )
        rtg_improves_max = torch.max(rtg_improves).item()
        if rtg_improves_max <= improve_threshold:
            return [episode, ]
        else:
            replace_pos = torch.argmax(rtg_improves).item()
            #print(state_pairs[replace_pos])
            i, j = epi_index[0][1][replace_pos]
            print(i,j)
            return_improve.append(rtg_improves_max)
            state_pairs_check.append(state_pairs[replace_pos].cpu().numpy())
            assert abs(epi_delta_rtg[replace_pos].item() - rewards[i:j].sum()) < 0.0001

            left, right = {}, {}
            left['observations'], right['observations'] = obss[:i+1], obss[j:]
            left['next_observations'], right['next_observations'] = obss_next[:i+1], obss_next[j:]
            left['actions'], right['actions'] = actions[:i+1], actions[j:]
            left['rewards'], right['rewards'] = rewards[:i+1], rewards[j:]
            left['terminals'], right['terminals'] = terminals[:i+1], terminals[j:]

            return solit_by_short_path(left, improve_threshold) + solit_by_short_path(right, improve_threshold)  # 对左右子序列分别进行递归，并将结果合并

    for epi_idx, epi in enumerate(episodes):
        return_improve = []
        state_pairs_check = []
        raw_return = epi['rewards'].sum().item()

        # 切分子轨迹，其中通过预测网络连接
        sub_epis = solit_by_short_path(epi, max_improve[epi_idx]*short_path_ratio)
        state_pairs = []
        for i in range(len(sub_epis)-1):
            state_pairs.append((sub_epis[i]['observations'][-1], sub_epis[i+1]['observations'][0]))

        if len(sub_epis) == 1:
            continue

        # 预测网络连接进行预测
        state_pairs = np.concatenate(state_pairs)      
        assert np.allclose(np.sort(state_pairs.flatten()), np.sort(np.array(state_pairs_check).flatten()))
        state_pairs = torch.tensor(state_pairs).reshape(-1, 2*config.model.DRR.state_dim).cuda()
        #print(state_pairs)
        pred_drs, pred_as = use_model_to_pred(model, state_pairs)
        if state_pairs.shape[0] == 1:
            pred_drs = pred_drs.unsqueeze(0)
            pred_as = pred_as.unsqueeze(0)
            
        # 构造带捷径的轨迹
        epi = sub_epis[0]
        for s_epi, dr, a in zip(sub_epis[1:], pred_drs, pred_as):
            epi['observations'] = np.concatenate((epi['observations'], s_epi['observations']))
            epi['next_observations'] = np.concatenate((epi['next_observations'], s_epi['next_observations']))
            epi['terminals'] = np.concatenate((epi['terminals'], s_epi['terminals']))
            epi['actions'][-1] = a.cpu().numpy()
            epi['rewards'][-1] = dr.item()
            epi['actions'] = np.concatenate((epi['actions'], s_epi['actions']))
            epi['rewards'] = np.concatenate((epi['rewards'], s_epi['rewards']))
            

        new_return = epi['rewards'].sum().item()
        print(f'episode {epi_idx}: improve {new_return - raw_return}')
        print(sum(return_improve))
    return episodes
'''

if __name__ == '__main__':
    env_name = 'hopper'
    data_type = 'medium'
    performance = 'len15-loss(dr)=0.066-loss(a)=0.014-MLPa32x64-MLPdr32x64'
    model_path = f'{base_path}/PredNet/ckpt/{env_name}-{data_type}/{performance}/model.pt'
    config_path = f'{base_path}/PredNet/ckpt/{env_name}-{data_type}/{performance}/config.json'

    # 加载参数
    config = load_config(config_path, env_name, data_type)

    # 构造模型对象
    model = PredNet(config.model)
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    # 加载原始轨迹
    trajectories = load_data(env_name, data_type)
    return_raw = [traj['rewards'].sum().item() for traj in trajectories]

    # 计算模型在轨迹内做 short_path 的最大提升  
    max_improve = get_max_improve(config, model, trajectories)

    # 设置所有捷径  
    episodes_shortpath = [] 

    '''
    short_path_ratios = [0.95,]
    for ratio in short_path_ratios:
        trajectories_raw = load_data(env_name, data_type)
        trajectories_raw = trajectories_raw[:20]
        episodes_shortpath.extend(bootstrap_short_path(config, model, max_improve, ratio, trajectories_raw))
        #episodes_shortpath.extend(no_overlap_short_path(config, model, max_improve, ratio, trajectories_raw))
    relabel_name = f'{env_name}-{data_type}-v2' + '(NO'+'|'.join([str(int(ratio * 10)) for ratio in short_path_ratios]) + ')'
    '''
    bootstrap_times = [10,]
    for t in bootstrap_times:
        trajectories_raw = load_data(env_name, data_type)
        trajectories_shortpath = bootstrap_short_path(config, model, max_improve, 0, trajectories_raw, times=t)
        episodes_shortpath.extend(trajectories_shortpath)
    relabel_name = f'{env_name}-{data_type}-v2' + '(BOt'+'|'.join([str(int(t)) for t in bootstrap_times]) + ')'
    return_shortpath = [traj['rewards'].sum().item() for traj in episodes_shortpath]
    #print(np.array(return_shortpath) - np.array(return_raw))

    
    # 保存轨迹文件
    with open(f'{base_path}/data/{relabel_name}.pkl', 'wb') as f:
        pickle.dump(episodes_shortpath, f)
    with open(f'{base_path}/data/{relabel_name}_App.pkl', 'wb') as f:
        episodes_shortpath = trajectories + episodes_shortpath
        pickle.dump(episodes_shortpath, f)
    