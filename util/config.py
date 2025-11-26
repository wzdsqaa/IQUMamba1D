import os
import yaml
from typing import Dict, Any

class MambaConfig:
    def __init__(self, config_path: str, train: bool = True):

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No Config File: {config_path}")
        
        # Load Config
        self.cfg = yaml.safe_load(open(config_path, 'r'))
        self.model_config = self.cfg['model_config']
        
    
    def _load_enc_config(self):
        cfg = self.model_config
        self.input_channels = cfg['input_channels']
        self.n_stages = cfg['n_stages']
        self.features_per_stage = cfg['features_per_stage']
        self.kernel_sizes = cfg['kernel_sizes']
        self.strides = cfg['strides']
        self.n_conv_per_stage = cfg['n_conv_per_stage']
        self.num_classes = cfg['num_classes']
        self.n_conv_per_stage_decoder = cfg['n_conv_per_stage_decoder']
        self.deep_supervision = cfg['deep_supervision']
        
