import torch
import torch.nn as nn

def forward_loss(outputs, targets , MSE_ratio , MAE_ratio ):
    """
    outputs: Tensor [bs, 13]
    targets: Tensor [bs, 13]
    """
    bs, dim = outputs.shape
    loss_total = 0.0
    mse_total, mae_total = 0.0, 0.0

    for i in range(dim):
        out_i = outputs[:, i]
        tgt_i = targets[:, i]

        # --- MSE & MAE ---
        mse = nn.MSELoss(reduction='sum')(out_i, tgt_i)
        mae = nn.L1Loss(reduction='sum')(out_i, tgt_i)

        loss_i = MSE_ratio*mse + MAE_ratio*mae

        loss_total += loss_i
        mse_total += mse
        mae_total += mae

    return loss_total, mse_total / dim, mae_total / dim

def forward_dynamic_loss(outputs, targets , weights):
    """
    outputs: Tensor [bs, 13]
    targets: Tensor [bs, 13]
    """
    bs, dim = outputs.shape
    mse_list, mae_list = [], []

    for i in range(dim):
        out_i = outputs[:, i]
        tgt_i = targets[:, i]

        # --- MSE & MAE ---
        mse = nn.MSELoss(reduction='mean')(out_i, tgt_i)
        mae = nn.L1Loss(reduction='mean')(out_i, tgt_i)
        mse_list.append(mse)
        mae_list.append(mae)

    losses = mse_list + mae_list  # 26 个 loss

    total_loss = sum(w * l for w, l in zip(weights, losses))

    return total_loss,losses

def dynamic_weight_average(loss_values, loss_history, T=2.0):
    """
    DWA 动态加权函数
    loss_values: list[float] 当前 epoch 每个指标的平均 loss
    loss_history: torch.Tensor[num_losses, 2] 保存最近两轮的平均损失
    T: 温度系数
    返回: weights, loss_history
    """
    num_losses = len(loss_values)
    loss_values = torch.tensor(loss_values)

    # 右移历史
    loss_history[:, 1] = loss_history[:, 0]
    loss_history[:, 0] = loss_values

    # 前两轮用均匀权重
    if (loss_history[:, 1] == 0).any():
        weights = torch.ones(num_losses)
    else:
        r = loss_history[:, 0] / (loss_history[:, 1] + 1e-8)
        weights = torch.softmax(r / T, dim=0) * num_losses

    return weights.detach(), loss_history
