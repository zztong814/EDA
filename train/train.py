import torch
from utils.loss import forward_loss,forward_dynamic_loss,dynamic_weight_average

def model_training(model, loader, optimizer, train_args,device):
    model.train()

    num_losses=train_args.train_DWA_num*2
    loss_history = torch.zeros((num_losses, 2))
    weights = torch.ones(num_losses) / num_losses

    for input, ground_truth,task_id,craft in loader:
        if(torch.cuda.is_available()):
            input, ground_truth,task_id,craft = input.to(device), ground_truth.to(device), task_id.to(device), craft.to(device)
        pred = model(input,craft, task_id)

        total_loss,losses = forward_dynamic_loss(pred, ground_truth,weights)
        # total_loss,_,_=forward_loss(pred, ground_truth, train_args.train_MSE_ratio, train_args.train_MAE_ratio)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        weights, loss_history = dynamic_weight_average(losses, loss_history, T=train_args.train_DWA_T)