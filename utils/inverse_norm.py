import torch

def inverse_norm(input_vec, task_id):
    batch_size, _ = input_vec.shape
    output_vec = torch.zeros_like(input_vec)
    for b in range(batch_size):
        tid = int(task_id[b].item())
        if tid == 1: # 5t正推，输入应该为5t电路的target
            output_vec[b][0] = output_vec[b][0] * (8883850.97442002 - 1316303.92528193) + 1316303.92528193
            output_vec[b][1] = output_vec[b][1] * (292.707702161998 - 0.02) + 0.02
            output_vec[b][2] = output_vec[b][2] * (100000000000 - 1885165.44) + 1885165.44
            output_vec[b][3] = output_vec[b][3] * (90.8545418680728 + 12.2475159224187) - 12.2475159224187
            output_vec[b][4] = output_vec[b][4] * (14038.8328670274 - 1.03019875662727) + 1.03019875662727
        elif tid == 2: # two_stage正推，输入应该为two_stage电路的target
            output_vec[b][0] = output_vec[b][0] * (109041361.776738 - 441507.434334005) + 441507.434334005
            output_vec[b][1] = output_vec[b][1] * (41554.1940092721 - 1.43669109021623) + 1.43669109021623
            output_vec[b][2] = output_vec[b][2] * (123495303.754714 - 64449.7274181243) + 64449.7274181243
            output_vec[b][3] = output_vec[b][3] * (149.918486082586 - 0.060702884639227) + 0.060702884639227
            output_vec[b][4] = output_vec[b][4] * (13243454.0487597 - 1.06801326426914) + 1.06801326426914
        elif tid == 3: # 5t反推，输入应该为5t电路的features
            output_vec[b][0:3] = input_vec[b][0:3] * (50 - 0.5) + 0.5
            output_vec[b][3:6] = input_vec[b][3:6] * (2 - 0.5) + 0.5
            output_vec[b][6] = input_vec[b][6] * (0.00002 - 0.000005) + 0.000005
        elif tid == 4: # two_stage反推，输入应该为two_stage电路的features
            output_vec[b][0:5] = input_vec[b][0:5] * (50 - 1) + 1
            output_vec[b][5:10] = input_vec[b][5:10] * (2 - 0.5) + 0.5
            output_vec[b][10] = input_vec[b][10] * (0.000000000002 - 0.0000000000001) + 0.0000000000001
            output_vec[b][11] = input_vec[b][11] * (100000 - 500) + 500
            output_vec[b][12] = input_vec[b][12] * (0.00001 - 0.00001) + 0.00001
        else:
            raise ValueError(f"非法 task_id {tid}")
    return output_vec