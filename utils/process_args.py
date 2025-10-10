# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

from dataclasses import dataclass, field
from typing import Optional, Tuple
import argparse
import transformers

@dataclass
class ModelArgs:
    model_d_model:           int = 128
    model_encoder_layers:    int=4
    model_d_ff:              int=512
    model_heads:             int=4
    model_dropout:           float=0.1

@dataclass
class TrainArgs:
    train_bs:            int = 32
    train_epochs:       int = 100
    train_lr:           float = 1e-3
    train_lr_min:       float = 1e-5
    train_weight_decay: float = 1e-2
    train_weight_B:     int = 1
    train_MSE_ratio:     float = 0.5
    train_MAE_ratio:    float = 0.5
    train_R2_ratio:     float = 0.0
    train_per_save_epochs: int = 10
    train_DWA_num:       int = 13
    train_DWA_T:            float = 2.0
    local_rank: int = field(default=0, metadata={"help": "For distributed training: local_rank"})

@dataclass
class EvalArgs:
    eval_bs:         int = 32
    eval_epochs_per_time: int=1
    eval_output_dir_A: str = 'output/eval/A'
    eval_output_dir_B: str = 'output/eval/B'
    eval_output_dir_C: str = 'output/eval/C'
    eval_output_dir_D: str = 'output/eval/D'

def process_args():
    parser = transformers.HfArgumentParser((ModelArgs, TrainArgs,EvalArgs))
    model_args, train_args ,eval_args = parser.parse_args_into_dataclasses()

    return model_args, train_args , eval_args

if __name__ == '__main__':
    model_args, train_args , eval_args=process_args()
    print(model_args)
    print(train_args)
    print(eval_args)
