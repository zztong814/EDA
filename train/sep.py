import torch
import os
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
import time

from dataset.load_data import get_dataset_four_models
from utils.process_args import process_args
from model.transformer2 import MultiInputTransformer
from model.mlp1 import Regressor
from train.train import model_training
from train.eval import model_evaluate,model_evaluate_task

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
    # model = MultiInputTransformer(d_model=model_args.model_d_model,
    #                               N=model_args.model_encoder_layers,
    #                               d_ff=model_args.model_d_ff,
    #                               h=model_args.model_heads,
    #                               dropout=model_args.model_dropout)

    model1 = Regressor()
    model2 = Regressor()
    model3 = Regressor()
    model4 = Regressor()
    if torch.cuda.is_available():
        model1 = model1.to(device)
        model2 = model2.to(device)
        model3 = model3.to(device)
        model4 = model4.to(device)

    # ====== 优化器 ======
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=train_args.train_lr, weight_decay=train_args.train_weight_decay)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=train_args.train_lr,
                                  weight_decay=train_args.train_weight_decay)
    optimizer3 = torch.optim.AdamW(model3.parameters(), lr=train_args.train_lr,
                                  weight_decay=train_args.train_weight_decay)
    optimizer4 = torch.optim.AdamW(model4.parameters(), lr=train_args.train_lr,
                                  weight_decay=train_args.train_weight_decay)
    scheduler_cosine1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=train_args.train_epochs, eta_min=train_args.train_lr_min)
    scheduler_cosine2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=train_args.train_epochs,
                                                                  eta_min=train_args.train_lr_min, verbose=True)
    scheduler_cosine3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=train_args.train_epochs,
                                                                  eta_min=train_args.train_lr_min, verbose=True)
    scheduler_cosine4 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer4, T_max=train_args.train_epochs,
                                                                  eta_min=train_args.train_lr_min, verbose=True)

    # ====== 主流程 ======
    start_time = time.time()
    print("=================Train Started=================")
    for epoch in tqdm(range(train_args.train_epochs), desc="Training Progress", ncols=100):
        model_training(model1, train_dataloader_A, optimizer1, train_args,device)
        model_training(model2, train_dataloader_B, optimizer2, train_args, device)
        model_training(model3, train_dataloader_C, optimizer3, train_args, device)
        model_training(model4, train_dataloader_D, optimizer4, train_args, device)

        if (epoch + 1) % eval_args.eval_epochs_per_time == 0:
            # evaluate task A
            total_fom_A = model_evaluate_task(model1, test_dataloader_A, epoch, eval_args.eval_output_dir_A, task="A",
                                              device=device)
            # evaluate task B
            total_fom_B = model_evaluate_task(model2, test_dataloader_B, epoch, eval_args.eval_output_dir_B, task="B",
                                              device=device)
            # evaluate task C
            total_fom_C = model_evaluate_task(model3, test_dataloader_C, epoch, eval_args.eval_output_dir_C, task="C",
                                              device=device)
            # evaluate task D
            total_fom_D = model_evaluate_task(model4, test_dataloader_D, epoch, eval_args.eval_output_dir_D, task="D",
                                              device=device)

            print(f"Eval Done: "
                  f"MSE_A = {total_fom_A:.6f},"
                  f"MSE_B = {total_fom_B:.6f},"
                  f"MSE_C = {total_fom_C:.6f},"
                  f"MSE_D = {total_fom_D:.6f}")

        scheduler_cosine1.step()
        scheduler_cosine2.step()
        scheduler_cosine3.step()
        scheduler_cosine4.step()

        train_time = time.time()

        # tqdm.write(f"Epoch {epoch+1}/{train_args.train_epochs} finished"
        #        f"Time={(train_time - start_time) // 60}min{(train_time - start_time) % 60}s")

    time.sleep(0.1)
    print("=================Train Finished=================")

if __name__ == '__main__':
    Joint_train()
