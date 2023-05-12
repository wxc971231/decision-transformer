import sys
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(base_path)

import torch
import wandb
import argparse
from pathlib import Path
import json
import os
from PredNet.code.data_Pred import StatePairDataset, create_dataset
from PredNet.code.utils_Pred import load_data, set_seed, create_empty_floder, CfgNode as CN
from PredNet.code.model_Pred import PredNet
from PredNet.code.trainer_Pred import Trainer


# -----------------------------------------------------------------------------
def get_config():
    C = CN()

    # system
    C.system = CN()
    #C.system.seeds = (3452, 3053, 3054, 3055, 3056)
    C.system.seeds = (3452, )
    
    # data
    C.data = StatePairDataset.get_default_config()

    # model
    C.model = PredNet.get_default_config()
    
    # trainer
    C.trainer = Trainer.get_default_config()
    return C

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    config = get_config()

    # custom setting
    save_wandb_log = True          # 启动 wanbd 记录训练曲线
    save_model = True              

    disable_all_log = False
    if disable_all_log:
        save_wandb_log = save_model = False

    # data setting
    config.data.env_name = 'hopper'
    config.data.data_type = 'medium'
    config.data.pred_len = 15
    config.data.split_test_set = True    
    trajectories = load_data(config.data.env_name, config.data.data_type)
    train_dataset, test_dataset = create_dataset(config.data, trajectories)

    # model setting
    config.model.BAR.hidden1_dim = 32
    config.model.BAR.hidden2_dim = 64
    config.model.BAR.action_dim = train_dataset.get_action_dim()
    config.model.BAR.state_dim = train_dataset.get_state_dim()
    config.model.DRR.hidden1_dim = 32
    config.model.DRR.hidden2_dim = 64
    config.model.DRR.state_dim = train_dataset.get_state_dim()

    # trainer setting
    config.trainer.max_epoch = 500+1
    config.trainer.batch_size = 5000
    config.trainer.learning_rate_DRR = 0.0005
    config.trainer.learning_rate_BAR = 0.0005  
    config.trainer.temperature_DRR = 1.0
    config.trainer.temperature_BAR = 1.0

    # iteration callback
    def epoch_end_callback(trainer):
        if trainer.epoch_num > 0 and trainer.epoch_num % 5 == 0:
            wandb.log({"epoch":trainer.epoch_num,
                       "train_loss_a": trainer.epoch_loss_a,
                       "train_loss_dr": trainer.epoch_loss_dr,})
            
            message = f"epoch_dt {trainer.epoch_dt * 1000:.2f}ms; epoch {trainer.epoch_num}: "
            message += f'{trainer.epoch_loss_a:.3f}(a) + {trainer.epoch_loss_dr:.3f}(dr)'
            print(message)

        if trainer.epoch_num >= 0 and trainer.epoch_num % 25 == 0:
            if config.data.split_test_set:
                eval_loss_a, eval_loss_dr = trainer.get_test_loss()
                #pred_loss_bdr_out, pred_act_acc_out = trainer.get_test_acc(max_distance=config.data.pred_len, map=map_for_test_pred_a)
                wandb.log({ "eval_loss_a":eval_loss_a,
                            "eval_loss_dr":eval_loss_dr,})
               
                message = f'{eval_loss_a:.3f}(a); {eval_loss_dr:.3f}(dr);\n'
                print(message)
            else:
                print(message)
        

        if trainer.epoch_num == config.trainer.max_epoch-1 and save_model:
            performance = f'len{config.data.pred_len}'
            performance += f'-loss(dr)={trainer.epoch_loss_dr:.3f}'
            performance += f'-loss(a)={trainer.epoch_loss_a:.3f}'
            performance += f"-MLPa{config.model.BAR.hidden1_dim}x{config.model.BAR.hidden2_dim}"
            performance += f"-MLPdr{config.model.DRR.hidden1_dim}x{config.model.DRR.hidden2_dim}"            
            
            floder_path = f'{base_path}/PredNet/ckpt/{config.data.env_name}-{config.data.data_type}/{performance}'
            create_empty_floder(floder_path)
            while True:
                try:
                    torch.save(trainer.model.state_dict(), f'{floder_path}/model.pt')
                    with open(f'{floder_path}/config.json', 'w') as f:
                        f.write(json.dumps(config.to_dict(), indent=4))
                    break
                except FileNotFoundError:
                    pass
            
            #torch.save({
            #    'epoch': trainer.epoch_num,
            #    'model_state_dict': trainer.model.state_dict(),
            #    'optimizer_state_dict': trainer.optimizer.state_dict(),
            #    'loss': trainer.epoch_loss,
            #    }, PATH)

    # load checkpoint
    '''
    PATH = f'{base_path}/DT/ckpt/epoch20_ret-353.0_trainloss2.051.pt'
    load_ckpt(trainer, PATH)
    eval_return, return_list = get_returns(trainer.model, config.trainer.env, ret=50)
    print(eval_return, return_list)
    '''

    # wandb
    if not save_wandb_log:
        os.environ['WANDB_MODE'] = 'offline'
    base = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
    wandb_path = Path(f'{base}/wandb')
    for T in [1.0, 0.8, 0.6]:
        config.trainer.temperature_BAR = T
        config.trainer.temperature_DRR = T
    
        for seed in config.system.seeds:
            set_seed(seed)
            config.data.data_split_seed = seed
            train_dataset, test_dataset = create_dataset(config.data, trajectories)
            config.trainer.batch_size = min(150000, len(train_dataset))
            #config.trainer.batch_size = 128

            # get experient name by setting
            group_name = f'{config.data.env_name}-{config.data.data_type}'
            group_name += f"-len{config.data.pred_len}"
            group_name += f"-T{config.trainer.temperature_BAR}(A);{config.trainer.temperature_DRR}(R)"
            group_name += f"-lr{config.trainer.learning_rate_BAR}(A);{config.trainer.learning_rate_DRR}(R)"
            group_name += f"-MLPdr{config.model.DRR.hidden1_dim}x{config.model.DRR.hidden2_dim}"
            group_name += f"-MLPa{config.model.BAR.hidden1_dim}x{config.model.BAR.hidden2_dim}"

            # construct the model and trainer
            model = PredNet(config.model)
            trainer = Trainer(config, model, train_dataset, test_dataset)
            trainer.set_callback('on_epoch_end', epoch_end_callback)

            with wandb.init(
                # set the wandb project where this run will be logged
                project="PredNet-1",
                dir = wandb_path,
                group = group_name,
                name = f"seed_{seed}",
                # track hyperparameters and run metadata
                config={
                    "dataset":f'{config.data.env_name}-{config.data.data_type}',
                    "epochs": config.trainer.max_epoch,
                    "pred_len":config.data.pred_len,
                    "lr_a": config.trainer.learning_rate_BAR,
                    "T_a": config.trainer.temperature_BAR,
                    "architecture_BAR": f"MLP-{config.model.BAR.hidden1_dim}-{config.model.BAR.hidden2_dim}",
                    "lr_dr": config.trainer.learning_rate_DRR,
                    "T_dr": config.trainer.temperature_DRR,
                    "architecture_DRR": f"MLP-{config.model.DRR.hidden1_dim}-{config.model.DRR.hidden2_dim}",
                    "batch_size": config.trainer.batch_size,}):

                wandb.watch(model, log='all', log_freq=10)
                # run the optimization
                trainer.run()

    #torch.onnx.export(model, torch.randn(config.trainer.batch_size, 3*config.model.block_size, config.model.n_embd).to(model.device), "model.onnx")
    #wandb.save("model.onnx")
    wandb.finish()