import numpy as np
import torch
import torch.nn as nn
import os
import sys
import random
import json
from pathlib import Path
import shutil
from datetime import datetime
import argparse
sys.path.append('/root/IQUMamba1D')  # Program's root path

from data_loader.dataloader import create_data_loaders
from calflops import calculate_flops
from util.logger import create_logger
from util.evaluation import test_model
from util.training import train_model
from util.utils import Create_Mamba_model, create_new_results_folder
from util.config import MambaConfig
from util.loss import si_snr_loss


def set_random_seeds(seed):
    """Set all random seeds to ensure reproducibility of experiments"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic algorithms
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seeds set to: {seed}")


def save_experiment_config(args, results_folder):
    """Save experiment configuration to JSON file"""
    config = {
        'timestamp': datetime.now().isoformat(),
        'data_choice': args.data_choice,
        'mode': args.mode,
        'loss_fun': args.loss_fun,
        'stage': args.stage,
        'seed': args.seed,
        'run_id': args.run_id if hasattr(args, 'run_id') else None,
        'multiple_runs': args.multiple_runs if hasattr(args, 'multiple_runs') else False
    }
    
    config_path = f'/root/IQUMamba1D/results/{results_folder}/config/experiment_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config


def get_model_config_path(stage):
    """Get configuration file path based on stage"""
    config_mapping = {
        2: '/root/IQUMamba1D/config/model_config_IQ_stage2.yaml',
        3: '/root/IQUMamba1D/config/model_config_IQ.yaml',
        4: '/root/IQUMamba1D/config/model_config_IQ_stage4.yaml',
        5: '/root/IQUMamba1D/config/model_config_IQ_stage5.yaml',
        8192: '/root/IQUMamba1D/config/model_config_IQ_stage4_8192.yaml',
        16384: '/root/IQUMamba1D/config/model_config_IQ_stage4_16384.yaml',
        32768: '/root/IQUMamba1D/config/model_config_IQ_stage4_32768.yaml'
    }
    

    return config_mapping.get(stage, config_mapping[3])



def setup_data_parameters(data_choice, logger):
    """Setup data-related parameters"""
    data_configs = {
        '2016': {'input_size': 128, 'num_points': 128, 'input_channels': 2},
        '2018': {'input_size': 1024, 'num_points': 256, 'input_channels': 2},
        'TorchSig': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK_M': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK_M_8192': {'input_size': 8192, 'num_points': 256, 'input_channels': 2},
        '8PSK_M_16384': {'input_size': 16384, 'num_points': 256, 'input_channels': 2},
        '8PSK_M_32768': {'input_size': 32768, 'num_points': 256, 'input_channels': 2},
        'QPSK_16APSK': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '8PSK_Rs': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '16QAM_64QAM': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '64QAM_64QAM': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '64QAM_128QAM': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
        '16QAM_64QAM_128QAM': {'input_size': 4096, 'num_points': 256, 'input_channels': 2},
    }
    
    if data_choice not in data_configs:
        raise ValueError(f"Unsupported data choice: {data_choice}")
    
    config = data_configs[data_choice]
    logger.info(f"Dataset: {data_choice}, DataLength: {config['input_size']}")
    
    return config['input_size'], config['num_points'], config['input_channels']


def create_model(args, cfg, input_size, device, logger):
    """Create and return model"""

    model = Create_Mamba_model(cfg, logger, input_size_=input_size)
    
    return model


def calculate_model_complexity(model, batch_size, input_channels, input_size, logger):
    """Calculate model complexity"""
    input_tuple = (batch_size, input_channels, input_size)
    flops, macs, params = calculate_flops(model, input_tuple, print_detailed=False)
    logger.info(f"InputSize: {input_size}, FLOPs: {flops}, MACs: {macs}, Params: {params}")
    return flops, macs, params


def setup_training_components(args, model, logger):
    """Setup training components (loss function, optimizer, scheduler)"""
    # Loss function
    if args.loss_fun == 'Huber':
        criterion = nn.HuberLoss()
        logger.info("Loss Function: HuberLoss")
    if args.loss_fun == 'SI-SNR':
        criterion = si_snr_loss
        logger.info("Loss Function: SI-SNR")
    else:  # MSE
        criterion = nn.MSELoss()
        logger.info("Loss Function: MSE")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=4,
        verbose=True
    )
    
    return criterion, optimizer, scheduler


def test_data_loading(args, logger):
    """Test only data loading functionality"""
    logger.info("=" * 50)
    logger.info("Data Loading Test Mode")
    logger.info("=" * 50)
    
    # Setup data parameters
    input_size, num_points, input_channels = setup_data_parameters(args.data_choice, logger)
    
    # Constant settings
    SIGNAL_NAMES = args.source_names
    NUM_SOURCES = len(SIGNAL_NAMES)
    batch_size = 32
    
    logger.info(f"Signal Names: {SIGNAL_NAMES}")
    logger.info(f"Number of Sources: {NUM_SOURCES}")
    logger.info(f"Input Size: {input_size}")
    logger.info(f"Input Channels: {input_channels}")
    logger.info(f"Batch Size: {batch_size}")
    
    try:
        # Create data loaders
        logger.info("Starting to create data loaders...")
        train_loader, val_loader, snr_loaders = create_data_loaders(
            batch_size,
            data_choice=args.data_choice,
            num_sources=NUM_SOURCES
        )
        
        logger.info("✓ Data loaders created successfully!")
        
        # Test training data loader
        logger.info("Testing training data loader...")
        train_iter = iter(train_loader)
        train_batch = next(train_iter)
        logger.info(f"✓ Training batch shapes: {[x.shape for x in train_batch]}")
        
        # Test validation data loader
        logger.info("Testing validation data loader...")
        val_iter = iter(val_loader)
        val_batch = next(val_iter)
        logger.info(f"✓ Validation batch shapes: {[x.shape for x in val_batch]}")
        
        # Test SNR data loaders
        logger.info("Testing SNR data loaders...")
        for snr, snr_loader in snr_loaders.items():
            snr_iter = iter(snr_loader)
            snr_batch = next(snr_iter)
            logger.info(f"✓ SNR {snr}dB batch shapes: {[x.shape for x in snr_batch]}")
        
        logger.info("=" * 50)
        logger.info("All data loading tests passed!")
        logger.info("=" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"Data loading test failed: {str(e)}")
        logger.error("=" * 50)
        raise e


def run_single_experiment(args, seed=None):
    """Run single experiment"""
    # Set random seeds
    if seed is not None:
        set_random_seeds(seed)
        args.seed = seed
    
    # If in data test mode, create simple temporary logger
    if args.mode == 'test_data':
        # Create console logger
        import logging
        logger = logging.getLogger('data_test')
        logger.setLevel(logging.INFO)
        
        # If no handlers, add console handler
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Execute data loading test
        test_data_loading(args, logger)
        return "data_test_completed"
    
    # Below is the original complete experiment flow (for train, test, BER modes)
    # Create results folder
    results_folder = create_new_results_folder()
    if seed is not None:
        results_folder = f"{results_folder}_seed_{seed}"
    
    # Create necessary folders
    folders = ['weights', 'logs', 'saved_plots', 'config']
    for folder in folders:
        os.makedirs(f'/root/IQUMamba1D/results/{results_folder}/{folder}', exist_ok=True)
    
    # Create logger
    logger = create_logger(f'/root/IQUMamba1D/results/{results_folder}/logs/output.log')
    logger.info(f"Starting experiment - Seed: {seed if seed is not None else 'None'}")
    
    # Save experiment configuration
    save_experiment_config(args, results_folder)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Setup data parameters
    input_size, num_points, input_channels = setup_data_parameters(args.data_choice, logger)
    
    # Get config file path and copy
    config_path = get_model_config_path(args.stage)
    shutil.copy2(config_path, f'/root/IQUMamba1D/results/{results_folder}/config')
    
    # Create config object
    cfg = MambaConfig(config_path, train=True)
    
    
    # Create model
    model = create_model(args, cfg, input_size, device, logger)
    
    # Calculate model complexity
    batch_size = 32
    flops, macs, params = calculate_model_complexity(model, batch_size, input_channels, input_size, logger)
    
    # Setup training components
    criterion, optimizer, scheduler = setup_training_components(args, model, logger)
    
    # Constant settings
    SIGNAL_NAMES = args.source_names
    NUM_SOURCES = len(SIGNAL_NAMES)
    num_epochs = 200
    early_stop_patience = 6
    
    # Execute corresponding mode
    if args.mode in ['train', 'test']:
        train_loader, val_loader, snr_loaders = create_data_loaders(
            batch_size,
            data_choice=args.data_choice,
            num_sources=NUM_SOURCES
        )
    
    if args.mode == 'train':
        logger.info("Training Mode")
        train_model(
            model, scheduler, train_loader, val_loader, snr_loaders,
            criterion, optimizer, device, num_epochs, early_stop_patience,
            logger, results_folder, data_choice=args.data_choice,
            num_plots=1, batch_size=batch_size, input_size=input_size,
            signal_names=SIGNAL_NAMES
        )

    
    elif args.mode == 'test':
        logger.info("Testing Mode")
        model.eval()
        state_dict = torch.load('/root/IQUMamba1D/checkpoint/best_model_weights.pth', weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Loaded model weights from /root/IQUMamba1D/checkpoint.")
        
        snr_metrics = test_model(
            model, snr_loaders, criterion, device, logger, results_folder,
            num_plots=1, num_points=num_points, input_size=input_size,
            data_choice=args.data_choice, signal_names=SIGNAL_NAMES
        )
        
        # Clean up checkpoint file
        checkpoint_path = '/root/IQUMamba1D/checkpoint/best_model_weights.pth'
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
    
    logger.info(f"Experiment completed - Results saved in: {results_folder}")
    return results_folder


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='IQU Mamba 1D Training and Testing Program')
    
    # Basic parameters

    parser.add_argument('--data_choice', type=str, default='8PSK_M',
                       choices=['TorchSig','2016','2018', '8PSK_M', '8PSK_M_8192', '8PSK_M_16384', '8PSK_M_32768', 'QPSK_16APSK', '8PSK_Rs', '16QAM_64QAM', '64QAM_64QAM', '64QAM_128QAM', '16QAM_64QAM_128QAM'],
                       help='Dataset: 2018/8PSK_M/8PSK_M_8192/8PSK_M_16384/8PSK_M_32768/QPSK_16APSK/8PSK_Rs/16QAM_64QAM/64QAM_64QAM/64QAM_128QAM/16QAM_64QAM_128QAM/16QAM_64QAM_128QAM_256QAM')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'test_data'],
                       help='Mode: train/test/test_data')
    parser.add_argument('--loss_fun', type=str, default='Huber',
                       choices=['MSE', 'Huber', 'SI-SNR'],
                       help='Loss function: MSE/Huber/SI-SNR')
    parser.add_argument('--stage', type=int, default=4,
                       choices=[2, 3, 4, 5, 6, 8192, 16384, 32768],
                       help='Stage number: 2/3/4/5/6/8192/16384/32768')
    parser.add_argument('--source_names', nargs='+', type=str)
    
    # Random seed related parameters
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (default not set)')
    parser.add_argument('--multiple_runs', action='store_true',
                       help='Perform multiple runs experiment')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='Number of runs for multiple runs (default 5)')
    parser.add_argument('--start_seed', type=int, default=42,
                       help='Starting seed for multiple runs (default 42)')
    
    # Other parameters
    parser.add_argument("--test_args", action="store_true",
                       help="Test command line arguments only, don't run actual experiment")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    if args.test_args:
        print(f"[Parameter testing mode]")
        print(f"Data: {args.data_choice}")
        print(f"Mode: {args.mode}")
        print(f"Signal Names: {args.source_names}")
        print(f"Seed: {args.seed}")
        print(f"Multiple Runs: {args.multiple_runs}")
        if args.multiple_runs:
            print(f"Number of Runs: {args.num_runs}")
            print(f"Starting Seed: {args.start_seed}")
        return
    
    print("=" * 80)
    print("IQU Mamba 1D Training and Testing Program")
    print("=" * 80)
    
    if args.multiple_runs:
        # Multiple runs experiment
        if args.mode == 'test_data':
            print(f"Data loading test mode - Performing {args.num_runs} tests...")
        else:
            print(f"Starting {args.num_runs} experiments...")
        
        results_folders = []
        
        for i in range(args.num_runs):
            current_seed = args.start_seed + i
            
            if args.mode == 'test_data':
                print(f"\n--- Data loading test {i+1}/{args.num_runs} (Seed: {current_seed}) ---")
            else:
                print(f"\n--- Running {i+1}/{args.num_runs} experiment (Seed: {current_seed}) ---")
            
            try:
                results_folder = run_single_experiment(args, seed=current_seed)
                results_folders.append(results_folder)
                
                if args.mode == 'test_data':
                    print(f"Data loading test {i+1} completed")
                else:
                    print(f"Experiment {i+1} completed: {results_folder}")
            except Exception as e:
                if args.mode == 'test_data':
                    print(f"Data loading test {i+1} failed: {str(e)}")
                else:
                    print(f"Experiment {i+1} failed: {str(e)}")
                continue
        
        if args.mode == 'test_data':
            print(f"\nAll data loading tests completed!")
        else:
            print(f"\nAll experiments completed! Result folders: {results_folders}")

    
    else:
        # Single run experiment
        if args.mode == 'test_data':
            if args.seed is not None:
                print(f"Running data loading test (Seed: {args.seed})")
            else:
                print("Running data loading test (No fixed seed)")
        else:
            if args.seed is not None:
                print(f"Running single experiment (Seed: {args.seed})")
            else:
                print("Running single experiment (No fixed seed)")
        
        try:
            results_folder = run_single_experiment(args, seed=args.seed)
            
            if args.mode == 'test_data':
                print(f"Data loading test completed!")
            else:
                print(f"Experiment completed! Results saved in: {results_folder}")
        except Exception as e:
            if args.mode == 'test_data':
                print(f"Data loading test failed: {str(e)}")
            else:
                print(f"Experiment failed: {str(e)}")
            raise


if __name__ == "__main__":
    main()