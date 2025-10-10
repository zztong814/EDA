import torch
import torch.nn as nn
import pandas as pd
import os
def model_evaluate(model, loader_task_A, loader_task_B,loader_task_C,loader_task_D,epoch, eval_args,device):

    #evaluate task A
    total_fom_A=model_evaluate_task(model, loader_task_A, epoch, eval_args.eval_output_dir_A,task="A",device=device)

    # evaluate task B
    total_fom_B=model_evaluate_task(model, loader_task_B, epoch, eval_args.eval_output_dir_B,task="B",device=device)

    # evaluate task C
    total_fom_C=model_evaluate_task(model, loader_task_C, epoch, eval_args.eval_output_dir_C,task="C",device=device)

    # evaluate task D
    total_fom_D=model_evaluate_task(model, loader_task_D, epoch, eval_args.eval_output_dir_D,task="D",device=device)

    print(f"Eval Done: "
          f"FoM_A = {total_fom_A:.6f},"
          f"FoM_B = {total_fom_B:.6f},"
          f"FoM_C = {total_fom_C:.6f},"
          f"FoM_D = {total_fom_D:.6f}")

def model_evaluate_task(model, loader, epoch , save_dir,task="A",device='cpu'):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, ground_truth,task_id,craft in loader:
            inputs, ground_truth = inputs.to(device), ground_truth.to(device)
            preds = model(inputs, craft, task_id)

            all_preds.append(preds.cpu())
            all_targets.append(ground_truth.cpu())

    # 拼接所有 batch -> [N, 13]
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    # 裁剪
    if task == "A"or task == "B":
        all_preds = all_preds[:, :5]
        all_targets = all_targets[:, :5]
    elif task == "C":
        all_preds = all_preds[:, :7]
        all_targets = all_targets[:, :7]
    else:
        all_preds = all_preds[:, :13]
        all_targets = all_targets[:, :13]

    n_targets = all_preds.shape[1]

    results = []
    total_fom = 0.0

    for i in range(n_targets):
        pred_i = all_preds[:, i]
        tgt_i = all_targets[:, i]

        mse = nn.MSELoss()(pred_i, tgt_i)
        mae = nn.L1Loss()(pred_i, tgt_i)

        if task == "A" or task == "B":
            ss_res = torch.sum((tgt_i - pred_i) ** 2)
            ss_tot = torch.sum((tgt_i - torch.mean(tgt_i)) ** 2)
            r2 = (1 - ss_res / (ss_tot + 1e-8))
            # fom = 0.3 * mse + 0.3 * mae + 0.4 * (1 - r2)
            fom = 0.5 * mse + 0.5 * mae
            results.append([mse, mae, r2, fom])
        else:
            fom = 0.5 * mse + 0.5 * mae
            results.append([mse, mae, fom])

        total_fom += fom

    # 保存到 Excel
    save_path=f"{save_dir}/{epoch}.csv"
    os.makedirs(save_dir, exist_ok=True)

    if task == "A" or task == "B":
        df = pd.DataFrame(
            results,
            columns=["MSE", "MAE", "R2", "FoM"],
            index=[f"Target_{i}" for i in range(n_targets)]
        )
    else:
        df = pd.DataFrame(
            results,
            columns=["MSE", "MAE", "FoM"],
            index=[f"Target_{i}" for i in range(n_targets)]
        )
    df.to_csv(save_path)

    return total_fom
