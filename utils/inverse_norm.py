import torch

def inverse_norm(input_vec, task_id):
    batch_size, _ = input_vec.shape
    output_vec = torch.zeros_like(input_vec)
    for b in range(batch_size):
        tid = int(task_id[b].item())
        if tid == 1: # 5t正推，输入应该为5t电路的target
            output_vec[b][0] = input_vec[b][0]/100 * (10602059.8211294 - 645632.611780252) + 645632.611780252
            output_vec[b][1] = input_vec[b][1]/100 * (292.707702161998 - 0.020728808400004) + 0.020728808400004
            output_vec[b][2] = input_vec[b][2]/100 * (100000000000 - 1221065.9825421) + 1221065.9825421
            output_vec[b][3] = input_vec[b][3]/100 * (149.258357674875 + 12.2475159224187) - 12.2475159224187
            output_vec[b][4] = input_vec[b][4]/100 * (1416090.54834304 - 1.03019875662727) + 1.03019875662727
        elif tid == 2: # two_stage正推，输入应该为two_stage电路的target
            output_vec[b][0] = input_vec[b][0]/100 * (262299959.900606 - 0) + 0
            output_vec[b][1] = input_vec[b][1]/100 * (41554.1940092721 - 0.006677072180007) + 0.006677072180007
            output_vec[b][2] = input_vec[b][2]/100 * (1e11 - 3791.34972457912) + 3791.34972457912
            output_vec[b][3] = input_vec[b][3]/100 * (179.989617050718 + 179.579391753632) - 179.579391753632
            output_vec[b][4] = input_vec[b][4]/100 * (13243454.0487597 - 1.06801326426914) + 1.06801326426914
        elif tid == 3: # 5t反推，输入应该为5t电路的features
            output_vec[b][0:3] = input_vec[b][0:3]/100 * (50 - 0.5) + 0.5
            output_vec[b][3:6] = input_vec[b][3:6]/100 * (2 - 0.5) + 0.5
            output_vec[b][6] = input_vec[b][6]/100* (0.00002 - 0.000005) + 0.000005
        elif tid == 4: # two_stage反推，输入应该为two_stage电路的features
            output_vec[b][0:5] = input_vec[b][0:5]/100 * (50 - 1) + 1
            output_vec[b][5:10] = input_vec[b][5:10] /100* (2 - 0.5) + 0.5
            output_vec[b][10] = input_vec[b][10]/100 * (0.000000000002 - 0.0000000000001) + 0.0000000000001
            output_vec[b][11] = input_vec[b][11]/100 * (100000 - 500) + 500
            output_vec[b][12] = input_vec[b][12]/100 * (0.00001 - 0.00001) + 0.00001
        else:
            raise ValueError(f"非法 task_id {tid}")
    return output_vec