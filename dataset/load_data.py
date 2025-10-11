import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from enum import Enum
import numpy as np

class TaskID(Enum):
    task_A = 1
    task_B = 2
    task_C = 3
    task_D = 4

class TechnologyType(Enum):
    source_tech = 1
    target_tech = 2

class OpampDataset(Dataset):
    def __init__(self, **dataframes):
        """
        Args:
            **dataframes: 任意数量的命名 DataFrame，如 design_features=df1, targets=df2
        """
        if not dataframes:
            raise ValueError("At least one DataFrame must be provided.")

        for name, df in dataframes.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"{name} is not a pandas DataFrame")
            setattr(self, name, df)

        # ---------- 长度一致性检查 ----------
        lengths = [len(df) for df in self.__dict__.values()]
        if len(set(lengths)) > 1:          # set 去重后长度 >1 说明不一致
            raise ValueError(
                f"All DataFrames must have the same number of rows. "
                f"Got {dict(zip(self.__dict__.keys(), lengths))}"
            )
        self._len = lengths[0]

    def __len__(self):
        """
        由于构造函数里已经进行了长度一致性检查，
        故可以随意选取一个datafram的长度作为Dataset的长度
        """
        return self._len

    def __getitem__(self, idx):
        """
        按既定顺序检查，有该dataframe则转换为tensor返回，没有则跳过
        """
        keys = ['input', 'output', 'task_id', 'technology_type']
        tensors = []
        for k in keys:
            if hasattr(self, k):
                df = getattr(self, k)
                # 对 1 列的 DataFrame 用 .values 会降维，用 .iloc[idx].values 即可
                tensors.append(
                    torch.tensor(df.iloc[idx].values, dtype=torch.float32)
                )
        return tuple(tensors)          # 长度 1~4，按上面顺序

def pad_to_13(df):
    """原地把 DataFrame 补到 13 列，缺几列就补几个 0 列。"""
    for i in range(df.shape[1], 13):
        df[i] = 0

def concat_dataset(A_input_df,B_input_df,C_input_df,D_input_df):
    A = A_input_df.copy()
    B = B_input_df.copy()
    C = C_input_df.copy()
    D = D_input_df.copy()

    A.columns = range(A.shape[1])
    B.columns = range(B.shape[1])
    C.columns = range(C.shape[1])
    D.columns = range(D.shape[1])

    output_df = pd.concat([A, B, C, D], ignore_index=True)
    return output_df

def load_validate_data(batch_size, shuffle, task_id:TaskID, technology_type:TechnologyType, file_path):
    features_for_test_df = pd.read_csv(file_path)
    pad_to_13(features_for_test_df)
    task_id_for_test_df = pd.DataFrame({'task_id': [task_id.value] * len(features_for_test_df)})
    technology_type_for_test_df = pd.DataFrame({'technology_type': [technology_type.value] * len(features_for_test_df)})
    validate_dataset = OpampDataset(input = features_for_test_df, task_id = task_id_for_test_df, technology_type = technology_type_for_test_df)
    validate_data_iter = DataLoader(validate_dataset, batch_size, shuffle)
    return validate_data_iter

def get_dataset(file_paths, target_technology_ratio):
    """
    file_path[:]依次输入的地址为
    [5t源工艺特征，5t源工艺性能，5t目标工艺特征，5t目标工艺性能，
    two_stage源工艺特征，two_stage源工艺性能，two_stage目标工艺特征，two_stage目标工艺性能]
    用法和load_data一样，除了参数去掉了batch_size, shuffle
    """
    source_features_5t_df = pd.read_csv(file_paths[0])
    source_targets_5t_df = pd.read_csv(file_paths[1])
    target_features_5t_df = pd.read_csv(file_paths[2])
    target_targets_5t_df = pd.read_csv(file_paths[3])
    source_features_two_stage_df = pd.read_csv(file_paths[4])
    source_targets_two_stage_df = pd.read_csv(file_paths[5])
    target_features_two_stage_df = pd.read_csv(file_paths[6])
    target_targets_two_stage_df = pd.read_csv(file_paths[7])
    pad_to_13(source_features_5t_df)
    pad_to_13(source_targets_5t_df)
    pad_to_13(target_features_5t_df)
    pad_to_13(target_targets_5t_df)
    pad_to_13(source_features_two_stage_df)
    pad_to_13(source_targets_two_stage_df)
    pad_to_13(target_features_two_stage_df)
    pad_to_13(target_targets_two_stage_df)
    # 根据目标工艺占比生成A、B、C、D四个任务对应的输入和输出
    # A任务（5t正推）
    A_k = int(len(target_features_5t_df) * target_technology_ratio)
    A_train_input_df = pd.concat([source_features_5t_df, target_features_5t_df.iloc[:A_k]], ignore_index=True)
    A_test_input_df = target_features_5t_df.iloc[A_k:].reset_index(drop=True)
    A_train_output_df = pd.concat([source_targets_5t_df, target_targets_5t_df.iloc[:A_k]], ignore_index=True)
    A_test_output_df = target_targets_5t_df.iloc[A_k:].reset_index(drop=True)
    # B任务（5t反推）
    B_train_input_df = A_train_output_df
    B_test_input_df = A_test_output_df
    B_train_output_df = A_train_input_df
    B_test_output_df = A_test_input_df
    # C任务（two_stage正推）
    C_k = int(len(target_features_two_stage_df) * target_technology_ratio)
    C_train_input_df = pd.concat([source_features_two_stage_df, target_features_two_stage_df.iloc[:C_k]], ignore_index=True)
    C_test_input_df = target_features_two_stage_df.iloc[C_k:].reset_index(drop=True)
    C_train_output_df = pd.concat([source_targets_two_stage_df, target_targets_two_stage_df.iloc[:C_k]], ignore_index=True)
    C_test_output_df = target_targets_two_stage_df.iloc[C_k:].reset_index(drop=True)
    # D任务（two_stage反推）
    D_train_input_df = C_train_output_df
    D_test_input_df = C_test_output_df
    D_train_output_df = C_train_input_df
    D_test_output_df = C_test_input_df
    # task_id
    A_train_task_id_df = pd.DataFrame({'task_id': [1] * (len(A_train_input_df))})
    A_test_task_id_df = pd.DataFrame({'task_id': [1] * (len(A_test_input_df))})
    B_train_task_id_df = pd.DataFrame({'task_id': [2] * (len(B_train_input_df))})
    B_test_task_id_df = pd.DataFrame({'task_id': [2] * (len(B_test_input_df))})
    C_train_task_id_df = pd.DataFrame({'task_id': [3] * (len(C_train_input_df))})
    C_test_task_id_df = pd.DataFrame({'task_id': [3] * (len(C_test_input_df))})
    D_train_task_id_df = pd.DataFrame({'task_id': [4] * (len(D_train_input_df))})
    D_test_task_id_df = pd.DataFrame({'task_id': [4] * (len(D_test_input_df))})
    # technology_type
    A_train_technology_type_df = pd.DataFrame({'technology_type': [1] * len(source_features_5t_df) + [2] * A_k})
    A_test_technology_type_df = pd.DataFrame({'technology_type': [2] * (len(target_features_5t_df) - A_k)})
    B_train_technology_type_df = A_train_technology_type_df
    B_test_technology_type_df = A_test_technology_type_df
    C_train_technology_type_df = pd.DataFrame({'technology_type': [1] * len(source_features_two_stage_df) + [2] * C_k})
    C_test_technology_type_df = pd.DataFrame({'technology_type': [2] * (len(target_features_two_stage_df) - C_k)})
    D_train_technology_type_df = C_train_technology_type_df
    D_test_technology_type_df = C_test_technology_type_df                                                         
    # 合并
    train_input_df=concat_dataset(A_train_input_df, B_train_input_df, C_train_input_df, D_train_input_df)
    train_output_df=concat_dataset(A_train_output_df, B_train_output_df, C_train_output_df, D_train_output_df)
    train_task_id_df=concat_dataset(A_train_task_id_df, B_train_task_id_df, C_train_task_id_df, D_train_task_id_df)
    train_technology_type_df=concat_dataset(A_train_technology_type_df, B_train_technology_type_df, C_train_technology_type_df, D_train_technology_type_df)
    # Dataset
    train_dataset = OpampDataset(input = train_input_df, output = train_output_df, task_id = train_task_id_df, technology_type = train_technology_type_df)
    A_test_dataset = OpampDataset(input = A_test_input_df, output = A_test_output_df, task_id = A_test_task_id_df, technology_type = A_test_technology_type_df)
    B_test_dataset = OpampDataset(input = B_test_input_df, output = B_test_output_df, task_id = B_test_task_id_df, technology_type = B_test_technology_type_df)
    C_test_dataset = OpampDataset(input = C_test_input_df, output = C_test_output_df, task_id = C_test_task_id_df, technology_type = C_test_technology_type_df)
    D_test_dataset = OpampDataset(input = D_test_input_df, output = D_test_output_df, task_id = D_test_task_id_df, technology_type = D_test_technology_type_df)
    return train_dataset, A_test_dataset, B_test_dataset, C_test_dataset, D_test_dataset

def load_train_and_test_data(batch_size, shuffle, file_paths, target_technology_ratio):
    """
    file_path[:]依次输入的地址为
    [5t源工艺特征，5t源工艺性能，5t目标工艺特征，5t目标工艺性能，
    two_stage源工艺特征，two_stage源工艺性能，two_stage目标工艺特征，two_stage目标工艺性能]
    """
    train_dataset, A_test_dataset, B_test_dataset, C_test_dataset, D_test_dataset = get_dataset(file_paths, target_technology_ratio)
    train_data_iter = DataLoader(train_dataset, batch_size, shuffle)
    A_test_data_iter = DataLoader(A_test_dataset, batch_size, shuffle)
    B_test_data_iter = DataLoader(B_test_dataset, batch_size, shuffle)
    C_test_data_iter = DataLoader(C_test_dataset, batch_size, shuffle)
    D_test_data_iter = DataLoader(D_test_dataset, batch_size, shuffle)
    # return
    return train_data_iter, A_test_data_iter, B_test_data_iter, C_test_data_iter, D_test_data_iter

def get_dataset_four_models(target_technology_ratio):
    p1 = 'dataset/dataset/01_train_set/5t_opamp/source/pretrain_design_features.csv'
    p2 = 'dataset/dataset/01_train_set/5t_opamp/source/pretrain_targets.csv'
    p3 = 'dataset/dataset/01_train_set/5t_opamp/target/target_design_features.csv'
    p4 = 'dataset/dataset/01_train_set/5t_opamp/target/target_targets.csv'
    p5 = 'dataset/dataset/01_train_set/two_stage_opamp/source/pretrain_design_features.csv'
    p6 = 'dataset/dataset/01_train_set/two_stage_opamp/source/pretrain_targets.csv'
    p7 = 'dataset/dataset/01_train_set/two_stage_opamp/target/target_design_features.csv'
    p8 = 'dataset/dataset/01_train_set/two_stage_opamp/target/target_targets.csv'
    source_features_5t_df = pd.read_csv(p1)
    source_targets_5t_df = pd.read_csv(p2)
    target_features_5t_df = pd.read_csv(p3)
    target_targets_5t_df = pd.read_csv(p4)
    source_features_two_stage_df = pd.read_csv(p5)
    source_targets_two_stage_df = pd.read_csv(p6)
    target_features_two_stage_df = pd.read_csv(p7)
    target_targets_two_stage_df = pd.read_csv(p8)
    pad_to_13(source_features_5t_df)
    pad_to_13(source_targets_5t_df)
    pad_to_13(target_features_5t_df)
    pad_to_13(target_targets_5t_df)
    pad_to_13(source_features_two_stage_df)
    pad_to_13(source_targets_two_stage_df)
    pad_to_13(target_features_two_stage_df)
    pad_to_13(target_targets_two_stage_df)
    # 根据目标工艺占比生成A、B、C、D四个任务对应的输入和输出
    # A任务（5t正推）
    A_k = int(len(target_features_5t_df) * target_technology_ratio)
    A_train_input_df = pd.concat([source_features_5t_df, target_features_5t_df.iloc[:A_k]], ignore_index=True)
    A_test_input_df = target_features_5t_df.iloc[A_k:].reset_index(drop=True)
    A_train_output_df = pd.concat([source_targets_5t_df, target_targets_5t_df.iloc[:A_k]], ignore_index=True)
    A_test_output_df = target_targets_5t_df.iloc[A_k:].reset_index(drop=True)
    # B任务（5t反推）
    B_train_input_df = A_train_output_df
    B_test_input_df = A_test_output_df
    B_train_output_df = A_train_input_df
    B_test_output_df = A_test_input_df
    # C任务（two_stage正推）
    C_k = int(len(target_features_two_stage_df) * target_technology_ratio)
    C_train_input_df = pd.concat([source_features_two_stage_df, target_features_two_stage_df.iloc[:C_k]], ignore_index=True)
    C_test_input_df = target_features_two_stage_df.iloc[C_k:].reset_index(drop=True)
    C_train_output_df = pd.concat([source_targets_two_stage_df, target_targets_two_stage_df.iloc[:C_k]], ignore_index=True)
    C_test_output_df = target_targets_two_stage_df.iloc[C_k:].reset_index(drop=True)
    # D任务（two_stage反推）
    D_train_input_df = C_train_output_df
    D_test_input_df = C_test_output_df
    D_train_output_df = C_train_input_df
    D_test_output_df = C_test_input_df
    # task_id
    A_train_task_id_df = pd.DataFrame({'task_id': [1] * (len(A_train_input_df))})
    A_test_task_id_df = pd.DataFrame({'task_id': [1] * (len(A_test_input_df))})
    B_train_task_id_df = pd.DataFrame({'task_id': [2] * (len(B_train_input_df))})
    B_test_task_id_df = pd.DataFrame({'task_id': [2] * (len(B_test_input_df))})
    C_train_task_id_df = pd.DataFrame({'task_id': [3] * (len(C_train_input_df))})
    C_test_task_id_df = pd.DataFrame({'task_id': [3] * (len(C_test_input_df))})
    D_train_task_id_df = pd.DataFrame({'task_id': [4] * (len(D_train_input_df))})
    D_test_task_id_df = pd.DataFrame({'task_id': [4] * (len(D_test_input_df))})
    # technology_type
    A_train_technology_type_df = pd.DataFrame({'technology_type': [1] * len(source_features_5t_df) + [2] * A_k})
    A_test_technology_type_df = pd.DataFrame({'technology_type': [2] * (len(target_features_5t_df) - A_k)})
    B_train_technology_type_df = A_train_technology_type_df
    B_test_technology_type_df = A_test_technology_type_df
    C_train_technology_type_df = pd.DataFrame({'technology_type': [1] * len(source_features_two_stage_df) + [2] * C_k})
    C_test_technology_type_df = pd.DataFrame({'technology_type': [2] * (len(target_features_two_stage_df) - C_k)})
    D_train_technology_type_df = C_train_technology_type_df
    D_test_technology_type_df = C_test_technology_type_df
    train_set_A = OpampDataset(input = A_train_input_df, output = A_train_output_df, task_id = A_train_task_id_df, technology_type = A_train_technology_type_df)
    train_set_B = OpampDataset(input = B_train_input_df, output = B_train_output_df, task_id = B_train_task_id_df, technology_type = B_train_technology_type_df)
    train_set_C = OpampDataset(input = C_train_input_df, output = C_train_output_df, task_id = C_train_task_id_df, technology_type = C_train_technology_type_df)
    train_set_D = OpampDataset(input = D_train_input_df, output = D_train_output_df, task_id = D_train_task_id_df, technology_type = D_train_technology_type_df)
    test_set_A = OpampDataset(input = A_test_input_df, output = A_test_output_df, task_id = A_test_task_id_df, technology_type = A_test_technology_type_df)
    test_set_B = OpampDataset(input = B_test_input_df, output = B_test_output_df, task_id = B_test_task_id_df, technology_type = B_test_technology_type_df)
    test_set_C = OpampDataset(input = C_test_input_df, output = C_test_output_df, task_id = C_test_task_id_df, technology_type = C_test_technology_type_df)
    test_set_D = OpampDataset(input = D_test_input_df, output = D_test_output_df, task_id = D_test_task_id_df, technology_type = D_test_technology_type_df)
    return train_set_A,train_set_B,train_set_C,train_set_D, test_set_A, test_set_B, test_set_C, test_set_D

def get_dataset_default(target_technology_ratio):
    p1 = 'dataset/dataset/01_train_set/5t_opamp/source/pretrain_design_features.csv'
    p2 = 'dataset/dataset/01_train_set/5t_opamp/source/pretrain_targets.csv'
    p3 = 'dataset/dataset/01_train_set/5t_opamp/target/target_design_features.csv'
    p4 = 'dataset/dataset/01_train_set/5t_opamp/target/target_targets.csv'
    p5 = 'dataset/dataset/01_train_set/two_stage_opamp/source/pretrain_design_features.csv'
    p6 = 'dataset/dataset/01_train_set/two_stage_opamp/source/pretrain_targets.csv'
    p7 = 'dataset/dataset/01_train_set/two_stage_opamp/target/target_design_features.csv'
    p8 = 'dataset/dataset/01_train_set/two_stage_opamp/target/target_targets.csv'
    path_list = [p1, p2, p3, p4, p5, p6, p7, p8]
    return get_dataset( path_list, target_technology_ratio)

# 测试代码，用前可以参考一下
if __name__ == "__main__":
    p1 = 'dataset/01_train_set/5t_opamp/source/pretrain_design_features.csv'
    p2 = 'dataset/01_train_set/5t_opamp/source/pretrain_targets.csv'
    p3 = 'dataset/01_train_set/5t_opamp/target/target_design_features.csv'
    p4 = 'dataset/01_train_set/5t_opamp/target/target_targets.csv'
    p5 = 'dataset/01_train_set/two_stage_opamp/source/pretrain_design_features.csv'
    p6 = 'dataset/01_train_set/two_stage_opamp/source/pretrain_targets.csv'
    p7 = 'dataset/01_train_set/two_stage_opamp/target/target_design_features.csv'
    p8 = 'dataset/01_train_set/two_stage_opamp/target/target_targets.csv'
#     p9 = '02_public_test_set/features/features_A.csv'
#     p10 = '02_public_test_set/features/features_B.csv'
#     p11 = '02_public_test_set/features/features_C.csv'
#     p12 = '02_public_test_set/features/features_D.csv'

#     A_test_data_iter = load_validate_data(10, True, TaskID.task_A, TechnologyType.target_tech, p9)
#     for input, task_id, technology_type in  A_test_data_iter:
#         print(input, task_id, technology_type)
#     B_test_data_iter = load_validate_data(10, True, TaskID.task_B, TechnologyType.target_tech, p10)
#     for input, task_id, technology_type in  A_test_data_iter:
#         print(input, task_id, technology_type)
#     C_test_data_iter = load_validate_data(10, True, TaskID.task_C, TechnologyType.target_tech, p11)
#     for input, task_id, technology_type in  A_test_data_iter:
#         print(input, task_id, technology_type)
#     D_test_data_iter = load_validate_data(10, True, TaskID.task_D, TechnologyType.target_tech, p12)
#     for input, task_id, technology_type in  A_test_data_iter:
#         print(input, task_id, technology_type)
    
    path_list = [p1,p2,p3,p4,p5,p6,p7,p8]
    train_set, test_set_A, test_set_B, test_set_C, test_set_D=get_dataset(path_list, 0.9)
    print(train_set[0])
#     train_data_iter, A_test_data_iter, B_test_data_iter, C_test_data_iter, D_test_data_iter = load_train_and_test_data(10, True, path_list, 0.9)
#     for input, output, task_id, technology_type in A_test_data_iter:
#         print(input, task_id, technology_type)
#     for input, output, task_id, technology_type in B_test_data_iter:
#         print(input, task_id, technology_type)
#     for input, output, task_id, technology_type in C_test_data_iter:
#         print(input, task_id, technology_type)
#     for input, output, task_id, technology_type in D_test_data_iter:
#         print(input, task_id, technology_type)
#     for input, output, task_id, technology_type in train_data_iter:
#         print(input, task_id, technology_type)