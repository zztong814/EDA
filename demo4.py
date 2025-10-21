import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import time

from dataset.load_data import get_dataset_four_models,get_dataset_seperate
from utils.process_args import process_args
from model.transformer2 import MultiInputTransformer
from tab_transformer_pytorch import FTTransformer
from model.mlp1 import Regressor
from train.eval_FT import model_evaluate
from utils.loss import forward_loss,forward_dynamic_loss,dynamic_weight_average

def model_training(model, loader, optimizer, train_args,device):
    model.train()

    num_losses=train_args.train_DWA_num
    loss_history = torch.zeros((num_losses, 2))
    weights = torch.ones(num_losses) / num_losses
    total_loss_all,total_mse,total_mae=0.0,0.0,0.0
    for input, ground_truth,task_id,craft in loader:
        x_categ = torch.stack([craft, task_id], dim=1).long().squeeze(dim=2)
        if(torch.cuda.is_available()):
            input, x_categ = input.to(device), x_categ.to(device)
        pred = model(x_categ,input)

        # total_loss,losses = forward_dynamic_loss(pred, ground_truth,weights)
        total_loss,mse_total,mae_total=forward_loss(pred, ground_truth, train_args.train_MSE_ratio, train_args.train_MAE_ratio)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        total_loss_all+=total_loss
        total_mse+=mse_total
        total_mae+=mae_total
        # weights, loss_history = dynamic_weight_average(losses, loss_history, T=train_args.train_DWA_T)
    print(total_loss_all,total_mse,total_mae)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args, train_args, eval_args = process_args()
    print("===================================================")
    print(model_args)
    print(train_args)
    print(eval_args)
    print("===================================================")

    (train_set_A, train_set_B, train_set_C, train_set_D,
     test_set_A, test_set_B, test_set_C, test_set_D) = get_dataset_seperate(train_data_ratio=0.9, normalize=True,is_pretrain=True)
    # ====== 训练集 ======
    train_set = ConcatDataset([
        # train_set_A,
        train_set_B
        # train_set_C,
        # train_set_D
    ])
    train_loader = DataLoader(train_set, batch_size=train_args.train_bs, shuffle=True)

    # ====== 测试集 ======
    test_dataloader_A = DataLoader(test_set_A, batch_size=eval_args.eval_bs, shuffle=True)
    test_dataloader_B = DataLoader(test_set_B, batch_size=eval_args.eval_bs, shuffle=True)
    test_dataloader_C = DataLoader(test_set_C, batch_size=eval_args.eval_bs, shuffle=True)
    test_dataloader_D = DataLoader(test_set_D, batch_size=eval_args.eval_bs, shuffle=True)

    # === 构建模型 ===
    # model = FTTransformer(
    #     categories=[2, 4],  # 类别特征的类别数列表
    #     num_continuous=13,  # 连续特征数量
    #     dim=64,  # token embedding维度
    #     dim_out=13,  # 输出维度
    #     depth=4,  # Transformer层数
    #     heads=8,  # 多头注意力数
    #     attn_dropout=0.1,
    #     ff_dropout=0.1
    # )
    model = FTTransformer(
        categories=[2, 4],  # 类别特征的类别数列表
        num_continuous=13,  # 连续特征数量
        dim=64,  # token embedding维度
        dim_out=13,  # 输出维度
        depth=4,  # Transformer层数
        heads=8,  # 多头注意力数
        attn_dropout=0.1,
        ff_dropout=0.1
    )
    if torch.cuda.is_available():
        model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.train_lr,
                                  weight_decay=train_args.train_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_args.train_epochs,
                                                                  eta_min=train_args.train_lr_min)
    # scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=5)

    start_time = time.time()
    print("=================Pretrain Started=================")
    for epoch in tqdm(range(train_args.train_pretrain_epochs), desc="Training Progress", ncols=100,disable=True):
        model_training(model, train_loader, optimizer, train_args, device)

        if (epoch + 1) % eval_args.eval_epochs_per_time == 0:
            mse_a,_,_,_=model_evaluate(model,
                           test_dataloader_A,
                           test_dataloader_B,
                           test_dataloader_C,
                           test_dataloader_D,
                           epoch,
                           eval_args,
                           device)

        # scheduler.step(mse_a)
        scheduler.step()
        train_time = time.time()

    print("=================Pretrain Finished=================")

    # torch.save(model.state_dict(), train_args.train_pretrain_pth)
    print(f"=================Model Saved:{train_args.train_pretrain_pth}================= ")


if __name__ == '__main__':
    main()