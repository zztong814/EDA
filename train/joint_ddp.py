import torch
import os
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
import time

from dataset.load_data import get_dataset_default
from utils.process_args import process_args
from model.transformer2 import MultiInputTransformer
from train.train import model_training
from train.eval import model_evaluate

def ddp_set_up():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['SLURM_NTASKS'])
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank,local_rank,world_size,device

def ddp_close_up():
    torch.distributed.destroy_process_group()

def Joint_train():
    # ====== 初始化 ======
    rank,local_rank,world_size,device = ddp_set_up()

    model_args, train_args , eval_args = process_args()

    train_set, test_set_A, test_set_B, test_set_C, test_set_D = get_dataset_default(0.9)
    # ====== 训练集 ======
    sampler_train = DistributedSampler(train_set,shuffle=True)
    train_loader = DataLoader(train_set, batch_size=train_args.train_bs, sampler=sampler_train)

    # ====== 测试集 ======
    sampler_test_A = DistributedSampler(test_set_A,shuffle=True)
    sampler_test_B = DistributedSampler(test_set_B,shuffle=True)
    sampler_test_C = DistributedSampler(test_set_C,shuffle=True)
    sampler_test_D = DistributedSampler(test_set_D,shuffle=True)
    test_dataloader_B=DataLoader(test_set_B, batch_size=eval_args.eval_bs, sampler=sampler_test_B)
    test_dataloader_C=DataLoader(test_set_C, batch_size=eval_args.eval_bs, sampler=sampler_test_C)
    test_dataloader_D=DataLoader(test_set_D, batch_size=eval_args.eval_bs, sampler=sampler_test_D)
    test_dataloader_A=DataLoader(test_set_A, batch_size=eval_args.eval_bs, sampler=sampler_test_A)

    # ====== 模型 ======
    model = MultiInputTransformer(d_model=model_args.model_d_model,
                                  N=model_args.model_encoder_layers,
                                  d_ff=model_args.model_d_ff,
                                  h=model_args.model_heads,
                                  dropout=model_args.model_dropout)
    if torch.cuda.is_available():
        model = model.to(device)
    print(f"[rank {rank}],local_rank={local_rank},device={torch.cuda.current_device()}")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # ====== 优化器 ======
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.train_lr, weight_decay=train_args.train_weight_decay)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_args.train_epochs, eta_min=train_args.train_lr_min,verbose=True)

    # ====== 主流程 ======
    start_time = time.time()
    if(rank==0):
        print("=================Train Started=================")
    for epoch in tqdm(range(train_args.train_epochs), desc="Training Progress", ncols=100):
        model_training(model, train_loader, optimizer, train_args,device)

        if (epoch + 1) % train_args.eval_epochs_per_time == 0:
            if dist.get_rank() == 0:  # 只在主进程保存
                #todo
                model_evaluate(model,
                               test_dataloader_A,
                               test_dataloader_B,
                               test_dataloader_C,
                               test_dataloader_D,
                               epoch,
                               eval_args,
                               device)

                print("model saved!")
                # torch.save({
                #     "epoch": epoch,
                #     "model": model.module.state_dict(),
                #     "optimizer": optimizer.state_dict(),
                #     "scheduler": scheduler_cosine.state_dict(),
                # }, f"../checkpoint/checkpoint_{epoch + 1}.pth")
        scheduler_cosine.step()

        train_time = time.time()

        if dist.get_rank() == 0:  # 只在主进程保存
            tqdm.write(f"Epoch {epoch+1}/{train_args.train_epochs} finished"
                   f"Time={(train_time - start_time) // 60}min{(train_time - start_time) % 60}s")

    ddp_close_up()
    if (rank == 0):
        print("=================Train Finished=================")
