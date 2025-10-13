import torch
import os
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
import time

from dataset.load_data import get_dataset_default,get_dataset_four_models
from utils.process_args import process_args
from model.transformer2 import MultiInputTransformer
from train.train import model_training
from train.eval import model_evaluate

def Joint_train():
    # ====== 初始化 ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args, train_args , eval_args = process_args()

    print("===================================================")
    print(model_args)
    print(train_args)
    print(eval_args)
    print("===================================================")

    (train_set_A,train_set_B,train_set_C,train_set_D,
     test_set_A, test_set_B, test_set_C, test_set_D) = get_dataset_four_models(0.9,True)
    # ====== 训练集 ======
    train_dataloader_A = DataLoader(train_set_A, batch_size=train_args.train_bs, shuffle=True)
    train_dataloader_B = DataLoader(train_set_B, batch_size=train_args.train_bs, shuffle=True)
    train_dataloader_C = DataLoader(train_set_C, batch_size=train_args.train_bs, shuffle=True)
    train_dataloader_D = DataLoader(train_set_D, batch_size=train_args.train_bs, shuffle=True)

    # ====== 测试集 ======
    test_dataloader_A = DataLoader(test_set_A, batch_size=eval_args.eval_bs, shuffle=True)
    test_dataloader_B=DataLoader(test_set_B, batch_size=eval_args.eval_bs, shuffle=True)
    test_dataloader_C=DataLoader(test_set_C, batch_size=eval_args.eval_bs, shuffle=True)
    test_dataloader_D=DataLoader(test_set_D, batch_size=eval_args.eval_bs, shuffle=True)


    # ====== 模型 ======
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

    # ====== 优化器 ======
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.train_lr, weight_decay=train_args.train_weight_decay)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_args.train_epochs, eta_min=train_args.train_lr_min,verbose=True)

    # ====== 主流程 ======
    start_time = time.time()
    print("=================Train Started=================")
    for epoch in tqdm(range(train_args.train_epochs), desc="Training Progress", ncols=100):
        model_training(model, train_dataloader_A, optimizer, train_args,device)
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

            print("model saved!")
                # torch.save({
                #     "epoch": epoch,
                #     "model": model.module.state_dict(),
                #     "optimizer": optimizer.state_dict(),
                #     "scheduler": scheduler_cosine.state_dict(),
                # }, f"../checkpoint/checkpoint_{epoch + 1}.pth")
        scheduler_cosine.step()

        train_time = time.time()

        tqdm.write(f"Epoch {epoch+1}/{train_args.train_epochs} finished"
               f"Time={(train_time - start_time) // 60}min{(train_time - start_time) % 60}s")

    time.sleep(0.1)
    print("=================Train Finished=================")

if __name__ == '__main__':
    Joint_train()
