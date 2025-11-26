import os
import torch.nn as nn
import torch
from models.IQUMamba1D import IQUMamba1D
from torch.nn import LeakyReLU, InstanceNorm1d
from util.config import MambaConfig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_nn_module(module_name: str):
    module_map = {
        "LeakyReLU": LeakyReLU,
        "InstanceNorm1d": InstanceNorm1d,
    }
    return module_map.get(module_name, None)

def Create_Mamba_model(config: MambaConfig, logger, input_size_):
    global input_size
    input_size = input_size_

    config._load_enc_config()
    return _create_enc_model(config)



def _create_enc_model(config):
    return IQUMamba1D(
        input_size=input_size,
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        deep_supervision=config.deep_supervision,
    ).to(device)


import os
import glob

def create_new_results_folder(base_dir='results'):
    results_path = '/root/IQUMamba1D/results'
    os.makedirs(results_path, exist_ok=True)
    
    next_num = 0
    while True:
        base_pattern = os.path.join(results_path, f'{base_dir}_{next_num}*')
        if not glob.glob(base_pattern):
            break
        next_num += 1
    
    return f'{base_dir}_{next_num}'
