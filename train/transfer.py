import torch
import os
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
import time

from dataset.load_data import get_dataset_four_models,get_dataset_seperate
from utils.process_args import process_args
from model.transformer2 import MultiInputTransformer
from model.mlp1 import Regressor
from train.train import model_training
from train.eval import model_evaluate

def transfer_pretrain(model_args, train_args, eval_args,device):
    (train_set_A, train_set_B, train_set_C, train_set_D,
     test_set_A, test_set_B, test_set_C, test_set_D) =get_dataset_seperate(train_data_ratio=0.9, normalize=True, is_pretrain=True)

    # ====== 训练集 ======
    train_dataloader_A = DataLoader(train_set_A, batch_size=train_args.train_bs, shuffle=True)
    train_dataloader_B = DataLoader(train_set_B, batch_size=train_args.train_bs, shuffle=True)
    train_dataloader_C = DataLoader(train_set_C, batch_size=train_args.train_bs, shuffle=True)
    train_dataloader_D = DataLoader(train_set_D, batch_size=train_args.train_bs, shuffle=True)

    # ====== 测试集 ======
    test_dataloader_A = DataLoader(test_set_A, batch_size=eval_args.eval_bs, shuffle=True)
    test_dataloader_B = DataLoader(test_set_B, batch_size=eval_args.eval_bs, shuffle=True)
    test_dataloader_C = DataLoader(test_set_C, batch_size=eval_args.eval_bs, shuffle=True)
    test_dataloader_D = DataLoader(test_set_D, batch_size=eval_args.eval_bs, shuffle=True)

    model = MultiInputTransformer(d_model=model_args.model_d_model,
                                  N=model_args.model_encoder_layers,
                                  d_ff=model_args.model_d_ff,
                                  h=model_args.model_heads,
                                  dropout=model_args.model_dropout)
    print("===================================================")
    print(model)
    print("===================================================")
    if torch.cuda.is_available():
        model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.train_lr,
                                  weight_decay=train_args.train_weight_decay)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_args.train_epochs,
                                                                  eta_min=train_args.train_lr_min)

    start_time = time.time()
    print("=================Pretrain Started=================")
    for epoch in tqdm(range(train_args.train_pretrain_epochs), desc="Training Progress", ncols=100):
        model_training(model, train_dataloader_A, optimizer, train_args, device)
        model_training(model, train_dataloader_B, optimizer, train_args, device)
        model_training(model, train_dataloader_C, optimizer, train_args, device)
        model_training(model, train_dataloader_D, optimizer, train_args, device)

        if (epoch + 1) % eval_args.eval_epochs_per_time == 0:
            model_evaluate(model,
                           test_dataloader_A,
                           test_dataloader_B,
                           test_dataloader_C,
                           test_dataloader_D,
                           epoch,
                           eval_args,
                           device)

        scheduler_cosine.step()

        train_time = time.time()

        # tqdm.write(f"Epoch {epoch + 1}/{train_args.train_epochs} finished"
        #            f"Time={(train_time - start_time) // 60}min{(train_time - start_time) % 60}s")

    print("=================Pretrain Finished=================")
    # torch.save({
    #     "epoch": epoch,
    #     "model": model.module.state_dict(),
    #     "optimizer": optimizer.state_dict(),
    #     "scheduler": scheduler_cosine.state_dict(),
    # }, f"../checkpoint/checkpoint_{epoch + 1}.pth")
    torch.save(model.state_dict(), train_args.train_pretrain_pth)
    print(f"=================Model Saved:{train_args.train_pretrain_pth}================= ")

def transfer_finetune(model_args, train_args, eval_args,device):
    (train_set_A, train_set_B, train_set_C, train_set_D,
     test_set_A, test_set_B, test_set_C, test_set_D) = get_dataset_seperate(train_data_ratio=0.9, normalize=True, is_pretrain=False)

    # ====== 训练集 ======
    train_dataloader_A = DataLoader(train_set_A, batch_size=train_args.train_bs, shuffle=True)
    train_dataloader_B = DataLoader(train_set_B, batch_size=train_args.train_bs, shuffle=True)
    train_dataloader_C = DataLoader(train_set_C, batch_size=train_args.train_bs, shuffle=True)
    train_dataloader_D = DataLoader(train_set_D, batch_size=train_args.train_bs, shuffle=True)

    # ====== 测试集 ======
    test_dataloader_A = DataLoader(test_set_A, batch_size=eval_args.eval_bs, shuffle=True)
    test_dataloader_B = DataLoader(test_set_B, batch_size=eval_args.eval_bs, shuffle=True)
    test_dataloader_C = DataLoader(test_set_C, batch_size=eval_args.eval_bs, shuffle=True)
    test_dataloader_D = DataLoader(test_set_D, batch_size=eval_args.eval_bs, shuffle=True)

    model = MultiInputTransformer(d_model=model_args.model_d_model,
                                  N=model_args.model_encoder_layers,
                                  d_ff=model_args.model_d_ff,
                                  h=model_args.model_heads,
                                  dropout=model_args.model_dropout)
    print("===================================================")
    print(model)
    print("===================================================")
    if torch.cuda.is_available():
        model = model.to(device)

    model.load_state_dict(torch.load(train_args.train_pretrain_pth), strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.train_lr,
                                  weight_decay=train_args.train_weight_decay)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_args.train_epochs,
                                                                  eta_min=train_args.train_lr_min)

    start_time = time.time()
    print("=================Finetune Started=================")
    for epoch in tqdm(range(train_args.train_finetune_epochs), desc="Training Progress", ncols=100):
        model_training(model, train_dataloader_A, optimizer, train_args, device)
        model_training(model, train_dataloader_B, optimizer, train_args, device)
        model_training(model, train_dataloader_C, optimizer, train_args, device)
        model_training(model, train_dataloader_D, optimizer, train_args, device)

        if (epoch + 1) % eval_args.eval_epochs_per_time == 0:
            model_evaluate(model,
                           test_dataloader_A,
                           test_dataloader_B,
                           test_dataloader_C,
                           test_dataloader_D,
                           epoch,
                           eval_args,
                           device)

        scheduler_cosine.step()

        train_time = time.time()

        # tqdm.write(f"Epoch {epoch + 1}/{train_args.train_epochs} finished"
        #            f"Time={(train_time - start_time) // 60}min{(train_time - start_time) % 60}s")

    print("=================Finetune Finished=================")
    # torch.save({
    #     "epoch": epoch,
    #     "model": model.module.state_dict(),
    #     "optimizer": optimizer.state_dict(),
    #     "scheduler": scheduler_cosine.state_dict(),
    # }, f"../checkpoint/checkpoint_{epoch + 1}.pth")
    # torch.save(model.state_dict(), train_args.train_finetune_pth)
    print(f"=================Model Saved:{train_args.train_finetune_pth}================= ")

def transfer_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args, train_args, eval_args = process_args()
    print("===================================================")
    print(model_args)
    print(train_args)
    print(eval_args)
    print("===================================================")

    transfer_pretrain(model_args, train_args, eval_args, device)
    transfer_finetune(model_args, train_args, eval_args, device)

    print("=================Pipeline Finished=================")
