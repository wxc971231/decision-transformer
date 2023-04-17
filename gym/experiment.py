import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import os

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(exp_prefix:str, variant:dict,):
    device = variant.get('device', 'cuda')
    model_type = variant['model_type']
    log_to_wandb = variant.get('log_to_wandb', False)
    env_name, dataset = variant['env'], variant['dataset']
    
    # 环境、训练、测试参数
    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets 3600
        scale = 1000.               # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 6000] # 6000
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 2500]  # 5000
        scale = 1000.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]      # 50
        scale = 10.
    else:
        raise NotImplementedError
    
    if model_type == 'bc':
        env_targets = env_targets[:1]   # if testing bc, since BC ignores target, no need for different evaluations
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    K = variant['K']    # state、action 和 rtg 序列的 context 长度，如果长度不足则分别 pad 至这个长度
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']

    # load dataset
    dataset_path = f'/home/tim/桌面/git/decision-transformer/gym/data/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
    '''
    trajectories 是一个轨迹列表，每个轨迹是一个如下构成的字典
        trajectorie['observations']:        (traj_len, state_dim)    的 np.array 
        trajectorie['next_observations']:   (traj_len, state_dim)    的 np.array 
        trajectorie['actions']:             (traj_len, act_dim)      的 np.array 
        trajectorie['rewards']:             (traj_len, )             的 np.array 
        trajectorie['terminals'/'dones']:   (traj_len, )             的 np.array 
    '''
    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)
    num_timesteps = sum(traj_lens)  # 所有轨迹的总timestep

    # used for input normalization
    states = np.concatenate(states, axis=0) # (state总数, state维数)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    # 打印数据集信息
    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    # only train on top pct_traj trajectories (for %BC experiment)
    pct_traj = variant.get('pct_traj', 1.)              # 训练使用的 timestep 比例
    num_timesteps = max(int(pct_traj*num_timesteps), 1) # 总 timestep 至少为 1
    sorted_inds = np.argsort(returns)                   # 按 return 从 lowest to highest 排序轨迹
    num_trajectories = 1                                # 过滤后用于训练的轨迹数量
    timesteps, ind = traj_lens[sorted_inds[-1]], len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]                   # 过滤后留下用于训练的轨迹索引，按 return 从 lowest to highest 排序
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds]) # 设置轨迹被采样的概率和其含有timestep的数量成正比

    def get_batch(batch_size=256, max_len=K):
        # 按概率有放回地采样一个 batch
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]

            # 在轨迹 traj 中随机选一个位置 (timestep)，从此开始往后截取 max_len 长度的轨迹
            # 如果剩余不足max_len，则截取到轨迹末尾
            si = random.randint(0, traj['rewards'].shape[0] - 1)    

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))   # (1, <=max_len, state_dim)
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))          # (1, <=max_len, act_dim)
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))                # (1, <=max_len, 1)
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))             # (1, <=max_len, )
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))         # (1, slist_len, )
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1                   # 处理 timestep >= max_ep_len 的部分
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))  # (1, slist_len, ) 或 (1, slist_len+1, )
            if rtg[-1].shape[1] <= s[-1].shape[1]:                                      
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)        # (1, slist_len+1, 1) 补一个 0

            # 由于截取轨迹长度可能不足 max_len，进行 padding
            # padding and state + reward normalization
            tlen = s[-1].shape[1]       # 截取轨迹真实长度
            pad_len = max_len - tlen    # pad 长度
            s[-1] = np.concatenate([np.zeros((1, pad_len, state_dim)), s[-1]], axis=1)          # states    用全 0 向量在序列左侧 padding
            s[-1] = (s[-1] - state_mean) / state_std                                            # state normalization                                   
            a[-1] = np.concatenate([np.ones((1, pad_len, act_dim)) * -10., a[-1]], axis=1)      # actions   用全 -10 向量在序列左侧 padding
            r[-1] = np.concatenate([np.zeros((1, pad_len, 1)), r[-1]], axis=1)                  # rewards   用全 0 向量在序列左侧 padding
            d[-1] = np.concatenate([np.ones((1, pad_len)) * 2, d[-1]], axis=1)                  # terminals 用全 2 向量在序列左侧 padding
            rtg[-1] = np.concatenate([np.zeros((1, pad_len, 1)), rtg[-1]], axis=1) / scale      # rtgs      用全 0 向量在序列左侧 padding
            timesteps[-1] = np.concatenate([np.zeros((1, pad_len)), timesteps[-1]], axis=1)     # timesteps 用全 0 向量在序列左侧 padding
            mask.append(np.concatenate([np.zeros((1, pad_len)), np.ones((1, tlen))], axis=1))   # mask      用全 0 向量在序列左侧 padding（attn mask=0代表忽略）

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)              # (batch_size, max_len, state_dim)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)              # (batch_size, max_len, act_dim)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)              # (batch_size, max_len, 1)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)          # (batch_size, max_len+1, 1)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)                 # (batch_size, max_len, )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device) # (batch_size, max_len, ) 这是所有 token 的绝对 timestep
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)                             # (batch_size, max_len, )

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    group_name = f'{exp_prefix}-{env_name}-{dataset}' 
    for seed in [42, 43, 44]:

        # 模型对象
        if model_type == 'dt':
            model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=K,
                max_ep_len=max_ep_len,
                hidden_size=variant['embed_dim'],
                n_layer=variant['n_layer'],
                n_head=variant['n_head'],
                n_inner=4*variant['embed_dim'],
                activation_function=variant['activation_function'],
                n_positions=1024,
                resid_pdrop=variant['dropout'],
                attn_pdrop=variant['dropout'],
            )
        elif model_type == 'bc':
            model = MLPBCModel(
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=K,
                hidden_size=variant['embed_dim'],
                n_layer=variant['n_layer'],
            )
        else:
            raise NotImplementedError
        model = model.to(device=device)

        # 优化器和学习率调度器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=variant['learning_rate'],
            weight_decay=variant['weight_decay'],
        )
        warmup_steps = variant['warmup_steps']
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1)    # 在 warmup_steps 内，学习率从 0 线性增加到 1，然后固定为 1
        )

        # 训练器
        if model_type == 'dt':
            trainer = SequenceTrainer(
                model=model,
                optimizer=optimizer,
                batch_size=batch_size,
                get_batch=get_batch,
                scheduler=scheduler,
                loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
                eval_fns=[eval_episodes(tar) for tar in env_targets],
            )
        elif model_type == 'bc':
            trainer = ActTrainer(
                model=model,
                optimizer=optimizer,
                batch_size=batch_size,
                get_batch=get_batch,
                scheduler=scheduler,
                loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
                eval_fns=[eval_episodes(tar) for tar in env_targets],
            )

        # 记录训练数据         
        exp_prefix = f'{seed}'      # wandb log name 
        #exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'   # wandb log name

        if not log_to_wandb:
            os.environ['WANDB_MODE'] = 'offline'

        
        with wandb.init(
                project='decision-transformer',
                name=exp_prefix,
                group=group_name,
                config=variant
            ):
            # wandb.watch(model)  # wandb has some bug

            for iter in range(variant['max_iters']):
                outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
                if log_to_wandb:
                    wandb.log(outputs)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 环境名
    parser.add_argument('--env', type=str, default='walker2d')
    # 数据质量（medium, medium-replay, medium-expert, expert）
    parser.add_argument('--dataset', type=str, default='expert') 
    # 是否使用延时reward（normal for standard setting, delayed for sparse）
    parser.add_argument('--mode', type=str, default='normal')
    # state、action 和 rtg 序列的 context 长度，如果长度不足则分别 pad 至这个长度
    parser.add_argument('--K', type=int, default=20)
    # 训练使用的 top timestep 比例
    parser.add_argument('--pct_traj', type=float, default=1.)
    # batch_size
    parser.add_argument('--batch_size', type=int, default=64)
    # model type: dt for decision transformer, bc for behavior cloning
    parser.add_argument('--model_type', type=str, default='dt')  
    # token 嵌入维度
    parser.add_argument('--embed_dim', type=int, default=128)
    # transformer block 堆叠数量
    parser.add_argument('--n_layer', type=int, default=3)
    # attention head num
    parser.add_argument('--n_head', type=int, default=1)
    # transformer block 中最后 FFD MLP 使用的激活函数
    parser.add_argument('--activation_function', type=str, default='relu')
    # resid_pdrop 和 attn_pdrop
    parser.add_argument('--dropout', type=float, default=0.1)
    # AdamW 优化器使用的学习率
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    # AdamW 优化器使用的 weight_decay 系数
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    # 学习率 warmup 的步数
    parser.add_argument('--warmup_steps', type=int, default=10000)
    # 评估策略 return 时交互的轨迹数量
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    # 训练迭代轮数
    parser.add_argument('--max_iters', type=int, default=10)
    # 每轮训练迭代循环次数
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    # 训练设备
    parser.add_argument('--device', type=str, default='cuda')
    # log to wandb or not
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    
    args = parser.parse_args()
    experiment('gym-experiment', variant=vars(args)) # variant 是一个字典，形如 {'env': 'hopper', 'dataset': 'medium',...}