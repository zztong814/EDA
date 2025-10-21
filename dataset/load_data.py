import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

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

def min_max_normalization(df):
    for col in df.columns:
        if (df[col].min() == df[col].max()):
            df[col] = 100
        else:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())*100

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

def get_file_paths():
    """
    咱俩文件路径不一样，之后直接在这个文件里面修改容易点
    """
    p1 = 'dataset/dataset/01_train_set/5t_opamp/source/pretrain_design_features.csv'
    p2 = 'dataset/dataset/01_train_set/5t_opamp/source/pretrain_targets.csv'
    p3 = 'dataset/dataset/01_train_set/5t_opamp/target/target_design_features.csv'
    p4 = 'dataset/dataset/01_train_set/5t_opamp/target/target_targets.csv'
    p5 = 'dataset/dataset/01_train_set/two_stage_opamp/source/pretrain_design_features.csv'
    p6 = 'dataset/dataset/01_train_set/two_stage_opamp/source/pretrain_targets.csv'
    p7 = 'dataset/dataset/01_train_set/two_stage_opamp/target/target_design_features.csv'
    p8 = 'dataset/dataset/01_train_set/two_stage_opamp/target/target_targets.csv'
    path_list = [p1, p2, p3, p4, p5, p6, p7, p8]
    return path_list

def read_csv_to_df(file_paths, target_technology_ratio, normalize):
    """
    file_path[:]依次输入的地址为
    [5t源工艺特征,
     5t源工艺性能,
     5t目标工艺特征,
     5t目标工艺性能,
     two_stage源工艺特征,
     two_stage源工艺性能,
     two_stage目标工艺特征,
     two_stage目标工艺性能]
    """
    source_features_5t_df = pd.read_csv(file_paths[0])
    source_targets_5t_df = pd.read_csv(file_paths[1])
    target_features_5t_df = pd.read_csv(file_paths[2])
    target_targets_5t_df = pd.read_csv(file_paths[3])
    source_features_two_stage_df = pd.read_csv(file_paths[4])
    source_targets_two_stage_df = pd.read_csv(file_paths[5])
    target_features_two_stage_df = pd.read_csv(file_paths[6])
    target_targets_two_stage_df = pd.read_csv(file_paths[7])
    for df in [source_features_5t_df, 
               source_targets_5t_df, 
               target_features_5t_df,
               target_targets_5t_df,
               source_features_two_stage_df,
               source_targets_two_stage_df, 
               target_features_two_stage_df, 
               target_targets_two_stage_df]:
        # 进行归一化
        if normalize:
            min_max_normalization(df)
        # 补零至13列
        pad_to_13(df)

    # 根据目标工艺占比生成A、B、C、D四个任务对应的输入和输出
    # A任务（5t正推）
    A_k = int(len(target_features_5t_df) * target_technology_ratio)
    A_train_input_df = pd.concat([source_features_5t_df, target_features_5t_df.iloc[:A_k]], ignore_index=True)
    A_test_input_df = target_features_5t_df.iloc[A_k:].reset_index(drop=True)
    A_train_output_df = pd.concat([source_targets_5t_df, target_targets_5t_df.iloc[:A_k]], ignore_index=True)
    A_test_output_df = target_targets_5t_df.iloc[A_k:].reset_index(drop=True)
    # B任务（two_stage正推）
    B_k = int(len(target_features_two_stage_df) * target_technology_ratio)
    B_train_input_df = pd.concat([source_features_two_stage_df, target_features_two_stage_df.iloc[:B_k]], ignore_index=True)
    B_test_input_df = target_features_two_stage_df.iloc[B_k:].reset_index(drop=True)
    B_train_output_df = pd.concat([source_targets_two_stage_df, target_targets_two_stage_df.iloc[:B_k]], ignore_index=True)
    B_test_output_df = target_targets_two_stage_df.iloc[B_k:].reset_index(drop=True)
    # C任务（5t反推）
    C_train_input_df = A_train_output_df
    C_test_input_df = A_test_output_df
    C_train_output_df = A_train_input_df
    C_test_output_df = A_test_input_df
    # D任务（two_stage反推）
    D_train_input_df = B_train_output_df
    D_test_input_df = B_test_output_df
    D_train_output_df = B_train_input_df
    D_test_output_df = B_test_input_df
    # task_id
    A_train_task_id_df = pd.DataFrame({'task_id': [0] * (len(A_train_input_df))})
    A_test_task_id_df = pd.DataFrame({'task_id': [0] * (len(A_test_input_df))})
    B_train_task_id_df = pd.DataFrame({'task_id': [1] * (len(B_train_input_df))})
    B_test_task_id_df = pd.DataFrame({'task_id': [1] * (len(B_test_input_df))})
    C_train_task_id_df = pd.DataFrame({'task_id': [2] * (len(C_train_input_df))})
    C_test_task_id_df = pd.DataFrame({'task_id': [2] * (len(C_test_input_df))})
    D_train_task_id_df = pd.DataFrame({'task_id': [3] * (len(D_train_input_df))})
    D_test_task_id_df = pd.DataFrame({'task_id': [3] * (len(D_test_input_df))})
    # technology_type
    A_train_technology_type_df = pd.DataFrame({'technology_type': [1] * len(source_features_5t_df) + [2] * A_k})
    A_test_technology_type_df = pd.DataFrame({'technology_type': [2] * (len(target_features_5t_df) - A_k)})
    B_train_technology_type_df = pd.DataFrame({'technology_type': [1] * len(source_features_two_stage_df) + [2] * B_k})
    B_test_technology_type_df = pd.DataFrame({'technology_type': [2] * (len(target_features_two_stage_df) - B_k)})
    C_train_technology_type_df = A_train_technology_type_df
    C_test_technology_type_df = A_test_technology_type_df
    D_train_technology_type_df = B_train_technology_type_df
    D_test_technology_type_df = B_test_technology_type_df

    df_tuple = (A_train_input_df, B_train_input_df, C_train_input_df, D_train_input_df, 
                A_train_output_df, B_train_output_df, C_train_output_df, D_train_output_df, 
                A_train_task_id_df, B_train_task_id_df, C_train_task_id_df, D_train_task_id_df,
                A_train_technology_type_df, B_train_technology_type_df, C_train_technology_type_df, D_train_technology_type_df,
                A_test_input_df, B_test_input_df, C_test_input_df, D_test_input_df, 
                A_test_output_df, B_test_output_df, C_test_output_df, D_test_output_df, 
                A_test_task_id_df, B_test_task_id_df, C_test_task_id_df, D_test_task_id_df,
                A_test_technology_type_df, B_test_technology_type_df, C_test_technology_type_df, D_test_technology_type_df)
    
    return df_tuple

def get_dataset_one_model(target_technology_ratio, normalize):
    (A_train_input_df, B_train_input_df, C_train_input_df, D_train_input_df, 
     A_train_output_df, B_train_output_df, C_train_output_df, D_train_output_df, 
     A_train_task_id_df, B_train_task_id_df, C_train_task_id_df, D_train_task_id_df,
     A_train_technology_type_df, B_train_technology_type_df, C_train_technology_type_df, D_train_technology_type_df,
     A_test_input_df, B_test_input_df, C_test_input_df, D_test_input_df, 
     A_test_output_df, B_test_output_df, C_test_output_df, D_test_output_df, 
     A_test_task_id_df, B_test_task_id_df, C_test_task_id_df, D_test_task_id_df,
     A_test_technology_type_df, B_test_technology_type_df, C_test_technology_type_df, D_test_technology_type_df) = read_csv_to_df(get_file_paths(), target_technology_ratio, normalize)
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

def get_dataset_four_models(target_technology_ratio, normalize):
    (A_train_input_df, B_train_input_df, C_train_input_df, D_train_input_df, 
     A_train_output_df, B_train_output_df, C_train_output_df, D_train_output_df, 
     A_train_task_id_df, B_train_task_id_df, C_train_task_id_df, D_train_task_id_df,
     A_train_technology_type_df, B_train_technology_type_df, C_train_technology_type_df, D_train_technology_type_df,
     A_test_input_df, B_test_input_df, C_test_input_df, D_test_input_df, 
     A_test_output_df, B_test_output_df, C_test_output_df, D_test_output_df, 
     A_test_task_id_df, B_test_task_id_df, C_test_task_id_df, D_test_task_id_df,
     A_test_technology_type_df, B_test_technology_type_df, C_test_technology_type_df, D_test_technology_type_df) = read_csv_to_df(get_file_paths(), target_technology_ratio, normalize)
    # Dataset
    train_set_A = OpampDataset(input = A_train_input_df, output = A_train_output_df, task_id = A_train_task_id_df, technology_type = A_train_technology_type_df)
    train_set_B = OpampDataset(input = B_train_input_df, output = B_train_output_df, task_id = B_train_task_id_df, technology_type = B_train_technology_type_df)
    train_set_C = OpampDataset(input = C_train_input_df, output = C_train_output_df, task_id = C_train_task_id_df, technology_type = C_train_technology_type_df)
    train_set_D = OpampDataset(input = D_train_input_df, output = D_train_output_df, task_id = D_train_task_id_df, technology_type = D_train_technology_type_df)
    test_set_A = OpampDataset(input = A_test_input_df, output = A_test_output_df, task_id = A_test_task_id_df, technology_type = A_test_technology_type_df)
    test_set_B = OpampDataset(input = B_test_input_df, output = B_test_output_df, task_id = B_test_task_id_df, technology_type = B_test_technology_type_df)
    test_set_C = OpampDataset(input = C_test_input_df, output = C_test_output_df, task_id = C_test_task_id_df, technology_type = C_test_technology_type_df)
    test_set_D = OpampDataset(input = D_test_input_df, output = D_test_output_df, task_id = D_test_task_id_df, technology_type = D_test_technology_type_df)
    return train_set_A,train_set_B,train_set_C,train_set_D, test_set_A, test_set_B, test_set_C, test_set_D

def get_dataset_seperate(train_data_ratio, normalize, is_pretrain):
    file_paths = get_file_paths()
    source_features_5t_df = pd.read_csv(file_paths[0])
    source_targets_5t_df = pd.read_csv(file_paths[1])
    target_features_5t_df = pd.read_csv(file_paths[2])
    target_targets_5t_df = pd.read_csv(file_paths[3])
    source_features_two_stage_df = pd.read_csv(file_paths[4])
    source_targets_two_stage_df = pd.read_csv(file_paths[5])
    target_features_two_stage_df = pd.read_csv(file_paths[6])
    target_targets_two_stage_df = pd.read_csv(file_paths[7])
    for df in [source_features_5t_df, 
               source_targets_5t_df, 
               target_features_5t_df,
               target_targets_5t_df,
               source_features_two_stage_df,
               source_targets_two_stage_df, 
               target_features_two_stage_df, 
               target_targets_two_stage_df]:
        # 进行归一化
        if normalize:
            min_max_normalization(df)
        # 补零至13列
        pad_to_13(df)
    
    if is_pretrain:
        # A任务（5t正推）
        A_k = int(len(source_features_5t_df) * train_data_ratio)
        A_train_input_df = source_features_5t_df[ : A_k]
        A_test_input_df = source_features_5t_df[A_k : ]
        A_train_output_df = source_targets_5t_df[ : A_k]
        A_test_output_df = source_targets_5t_df[A_k : ]
        # B任务（two_stage正推）
        B_k = int(len(source_features_two_stage_df) * train_data_ratio)
        B_train_input_df = source_features_two_stage_df[ : B_k]
        B_test_input_df = source_features_two_stage_df[B_k : ]
        B_train_output_df = source_targets_two_stage_df[ : B_k]
        B_test_output_df = source_targets_two_stage_df[B_k : ]
        # C任务（5t反推）
        C_train_input_df = A_train_output_df
        C_test_input_df = A_test_output_df
        C_train_output_df = A_train_input_df
        C_test_output_df = A_test_input_df
        # D任务（two_stage反推）
        D_train_input_df = B_train_output_df
        D_test_input_df = B_test_output_df
        D_train_output_df = B_train_input_df
        D_test_output_df = B_test_input_df
        # technology_type
        A_train_technology_type_df = pd.DataFrame({'technology_type': [1] * len(A_train_input_df)})
        A_test_technology_type_df = pd.DataFrame({'technology_type': [1] * len(A_test_input_df)})
        B_train_technology_type_df = pd.DataFrame({'technology_type': [1] * len(B_train_input_df)})
        B_test_technology_type_df = pd.DataFrame({'technology_type': [1] * len(B_test_input_df)})
        C_train_technology_type_df = A_train_technology_type_df
        C_test_technology_type_df = A_test_technology_type_df
        D_train_technology_type_df = B_train_technology_type_df
        D_test_technology_type_df = B_test_technology_type_df
    else:
        # A任务（5t正推）
        A_k = int(len(target_features_5t_df) * train_data_ratio)
        A_train_input_df = target_features_5t_df[ : A_k]
        A_test_input_df = target_features_5t_df[A_k : ]
        A_train_output_df = target_targets_5t_df[ : A_k]
        A_test_output_df = target_targets_5t_df[A_k : ]
        # B任务（two_stage正推）
        B_k = int(len(target_features_two_stage_df) * train_data_ratio)
        B_train_input_df = target_features_two_stage_df[ : B_k]
        B_test_input_df = target_features_two_stage_df[B_k : ]
        B_train_output_df = target_targets_two_stage_df[ : B_k]
        B_test_output_df = target_targets_two_stage_df[B_k : ]
        # C任务（5t反推）
        C_train_input_df = A_train_output_df
        C_test_input_df = A_test_output_df
        C_train_output_df = A_train_input_df
        C_test_output_df = A_test_input_df
        # D任务（two_stage反推）
        D_train_input_df = B_train_output_df
        D_test_input_df = B_test_output_df
        D_train_output_df = B_train_input_df
        D_test_output_df = B_test_input_df
        # technology_type
        A_train_technology_type_df = pd.DataFrame({'technology_type': [2] * len(A_train_input_df)})
        A_test_technology_type_df = pd.DataFrame({'technology_type': [2] * len(A_test_input_df)})
        B_train_technology_type_df = pd.DataFrame({'technology_type': [2] * len(B_train_input_df)})
        B_test_technology_type_df = pd.DataFrame({'technology_type': [2] * len(B_test_input_df)})
        C_train_technology_type_df = A_train_technology_type_df
        C_test_technology_type_df = A_test_technology_type_df
        D_train_technology_type_df = B_train_technology_type_df
        D_test_technology_type_df = B_test_technology_type_df

    # task_id
    A_train_task_id_df = pd.DataFrame({'task_id': [0] * (len(A_train_input_df))})
    A_test_task_id_df = pd.DataFrame({'task_id': [0] * (len(A_test_input_df))})
    B_train_task_id_df = pd.DataFrame({'task_id': [1] * (len(B_train_input_df))})
    B_test_task_id_df = pd.DataFrame({'task_id': [1] * (len(B_test_input_df))})
    C_train_task_id_df = pd.DataFrame({'task_id': [2] * (len(C_train_input_df))})
    C_test_task_id_df = pd.DataFrame({'task_id': [2] * (len(C_test_input_df))})
    D_train_task_id_df = pd.DataFrame({'task_id': [3] * (len(D_train_input_df))})
    D_test_task_id_df = pd.DataFrame({'task_id': [3] * (len(D_test_input_df))})
        
    train_set_A = OpampDataset(input = A_train_input_df, output = A_train_output_df, task_id = A_train_task_id_df, technology_type = A_train_technology_type_df)
    train_set_B = OpampDataset(input = B_train_input_df, output = B_train_output_df, task_id = B_train_task_id_df, technology_type = B_train_technology_type_df)
    train_set_C = OpampDataset(input = C_train_input_df, output = C_train_output_df, task_id = C_train_task_id_df, technology_type = C_train_technology_type_df)
    train_set_D = OpampDataset(input = D_train_input_df, output = D_train_output_df, task_id = D_train_task_id_df, technology_type = D_train_technology_type_df)
    test_set_A = OpampDataset(input = A_test_input_df, output = A_test_output_df, task_id = A_test_task_id_df, technology_type = A_test_technology_type_df)
    test_set_B = OpampDataset(input = B_test_input_df, output = B_test_output_df, task_id = B_test_task_id_df, technology_type = B_test_technology_type_df)
    test_set_C = OpampDataset(input = C_test_input_df, output = C_test_output_df, task_id = C_test_task_id_df, technology_type = C_test_technology_type_df)
    test_set_D = OpampDataset(input = D_test_input_df, output = D_test_output_df, task_id = D_test_task_id_df, technology_type = D_test_technology_type_df)
    return train_set_A,train_set_B,train_set_C,train_set_D, test_set_A, test_set_B, test_set_C, test_set_D
                            
    
def get_val_dataset():
    input_A_df = source_features_5t_df = pd.read_csv('dataset/02_public_test_set/features/features_A.csv')
    input_B_df = source_features_5t_df = pd.read_csv('dataset/02_public_test_set/features/features_B.csv')
    input_C_df = source_features_5t_df = pd.read_csv('dataset/02_public_test_set/features/features_C.csv')
    input_D_df = source_features_5t_df = pd.read_csv('dataset/02_public_test_set/features/features_D.csv')
    # task_id
    task_id_A = pd.DataFrame({'task_id': [0] * (len(input_A_df))})
    task_id_B = pd.DataFrame({'task_id': [1] * (len(input_B_df))})
    task_id_C = pd.DataFrame({'task_id': [2] * (len(input_C_df))})
    task_id_D = pd.DataFrame({'task_id': [3] * (len(input_D_df))})
    # technology_type
    technology_type_A = pd.DataFrame({'technology_type': [2] * (len(input_A_df))})
    technology_type_B = pd.DataFrame({'technology_type': [2] * (len(input_B_df))})
    technology_type_C = pd.DataFrame({'technology_type': [2] * (len(input_C_df))})
    technology_type_D = pd.DataFrame({'technology_type': [2] * (len(input_D_df))})

    val_set_A = OpampDataset(input = input_A_df, task_id = task_id_A, technology_type = technology_type_A)
    val_set_B = OpampDataset(input = input_B_df, task_id = task_id_B, technology_type = technology_type_B)
    val_set_C = OpampDataset(input = input_C_df, task_id = task_id_C, technology_type = technology_type_C)
    val_set_D = OpampDataset(input = input_D_df, task_id = task_id_D, technology_type = technology_type_D)

    return val_set_A, val_set_B, val_set_C, val_set_D

    


# 测试代码，用前可以参考一下
if __name__ == "__main__":
    val_set_A, val_set_B, val_set_C, val_set_D = get_val_dataset()
    print(val_set_A[0])
    print(val_set_B[0])
    print(val_set_C[0])
    print(val_set_D[0])