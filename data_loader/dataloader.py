from collections import defaultdict
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Union

class BaseSignalDataset(Dataset, ABC):
    """Base class for signal datasets"""
    
    def __init__(self, num_sources: int = 2):
        """
        Args:
            num_sources: Number of source signals (2, 3, 4, ...)
        """
        self.num_sources = num_sources
        self.signals = []  # Store all source signals
        self.mixture = []  # Store mixed signals
        self.snrs = []     # Store SNR labels
        self.num_samples = 0
    
    @abstractmethod
    def _load_data(self, *args, **kwargs):
        """Abstract method for loading data"""
        pass
    
    def _extract_snr_from_path(self, path: str) -> float:
        """Extract SNR value from path"""
        match = re.search(r'SNR=([-\d.]+)dB', path)
        if match:
            return float(match.group(1))
        else:
            raise ValueError(f"Cannot extract SNR information from path {path}") 
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get mixed signal
        mixed_signal = self.mixture[idx]
        mixed_real, mixed_imag = mixed_signal[:, 0], mixed_signal[:, 1]
        input_signal = np.stack([mixed_real, mixed_imag], axis=0)
        
        # Get each source signal and organize as target format
        target_channels = []
        for source_idx in range(self.num_sources):
            signal = self.signals[source_idx][idx]
            signal_real, signal_imag = signal[:, 0], signal[:, 1]
            target_channels.extend([signal_real, signal_imag])
        
        target = np.stack(target_channels, axis=0)  # (2*num_sources, signal_length)
        snr = self.snrs[idx]
        
        return (
            torch.tensor(input_signal, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32), 
            snr
        )


class MATLABSignalDataset(BaseSignalDataset):
    """MATLAB generated signal dataset - supports multi-source data in same file"""
    
    def __init__(self, signal_paths: List[str], mixture_paths: List[str], 
                 data_choice: str, num_sources: int = 2):
        """
        Args:
            signal_paths: Target signal path list (each file contains all sources)
            mixture_paths: Mixed signal path list
            data_choice: Data type selection ('QAM', '8PSK', etc.)
            num_sources: Number of source signals
        """
        super().__init__(num_sources)
        self.data_choice = data_choice
        self._load_data(signal_paths, mixture_paths)
    
    def _load_data(self, signal_paths: List[str], mixture_paths: List[str]):
        """Load MATLAB data - separate multiple sources from same file"""
        # Determine data field name
        name_str = 'ideal_frames'
        mix_name_str = 'mixed_frames'
        
        # Initialize source signal list
        for _ in range(self.num_sources):
            self.signals.append([])
        
        # Load target signals and separate sources
        for path in signal_paths:
            with h5py.File(path, 'r') as f:
                data = f[name_str][:]
                data = np.transpose(data, (2, 1, 0))  # (B, L, 2*num_sources)
            
            # Check if dimensions are correct
            expected_channels = 2 * self.num_sources
            if data.shape[2] != expected_channels:
                raise ValueError(
                    f"Channel count mismatch in file {path}: "
                    f"Expected {expected_channels} (2*{self.num_sources}), "
                    f"Actual {data.shape[2]}"
                )
            
            # Separate each source
            for source_idx in range(self.num_sources):
                # Extract real and imaginary parts of current source
                real_idx = source_idx * 2
                imag_idx = source_idx * 2 + 1
                source_data = data[:, :, [real_idx, imag_idx]]  # (B, L, 2)
                self.signals[source_idx].append(source_data)
            
            # Extract SNR information
            snr = self._extract_snr_from_path(path)
            self.snrs.extend([snr] * len(data))
        
        # Merge data from each source
        for source_idx in range(self.num_sources):
            self.signals[source_idx] = np.concatenate(self.signals[source_idx], axis=0)
        
        # Load mixed signals
        mixture_signals = []
        for path in mixture_paths:
            with h5py.File(path, 'r') as f:
                data = f[mix_name_str][:]
                data = np.transpose(data, (2, 1, 0))  # (B, L, 2)
            mixture_signals.append(data)
        
        self.mixture = np.concatenate(mixture_signals, axis=0)
        self.num_samples = len(self.mixture)
        
        print(f"Successfully loaded MATLAB dataset:")
        print(f"  Number of samples: {self.num_samples}")
        print(f"  Number of sources: {self.num_sources}")
        print(f"  Target signal shapes: {[signals.shape for signals in self.signals]}")
        print(f"  Mixed signal shape: {self.mixture.shape}")


class PublicSignalDataset(BaseSignalDataset):
    """Public datasets (RML2016/2018, TorchSig, etc.)"""
    
    def __init__(self, signal_paths: Dict[str, List[str]], data_choice: str, 
                 num_sources: int = 2):
        """
        Args:
            signal_paths: Dictionary of signal paths for each type {'BPSK': [paths], 'QPSK': [paths], ...}
            data_choice: Data type selection ('2016', '2018', 'TorchSig')
            num_sources: Number of source signals
        """
        super().__init__(num_sources)
        self.data_choice = data_choice
        self._load_data(signal_paths)
    
    def _load_data(self, signal_paths: Dict[str, List[str]]):
        """Load public dataset"""
        # Determine data field name
        field_mapping = {
            '2018': 'name',
            'TorchSig': 'data', 
            '2016': 'MAT'
        }
        name_str = field_mapping.get(self.data_choice)
        if not name_str:
            raise NotImplementedError(f'Unimplemented data type: {self.data_choice}')
        
        # Load signals of each type
        signal_types = sorted(signal_paths.keys())[:self.num_sources]
        
        for signal_type in signal_types:
            type_signals = []
            type_snrs = []
            
            for path in signal_paths[signal_type]:
                data = loadmat(path)[name_str]
                if self.data_choice == '2016':
                    data = 100 * np.transpose(data, (0, 2, 1))  # (1000,2,128) -> (1000,128,2)
                
                type_signals.append(data)
                snr = self._extract_snr_from_path(path)
                type_snrs.extend([snr] * len(data))
            
            self.signals.append(np.concatenate(type_signals, axis=0))
            if not self.snrs:
                self.snrs = type_snrs
        
        # Generate mixed signals (public dataset needs artificial mixing)
        self._generate_mixture()
    
    def _generate_mixture(self):
        """Generate mixed signals"""
        # Ensure all source signals have same number
        min_samples = min(len(signals) for signals in self.signals)
        self.num_samples = min_samples
        
        # Truncate all signals to same length
        for i in range(len(self.signals)):
            self.signals[i] = self.signals[i][:min_samples]
        
        self.snrs = self.snrs[:min_samples]
        
        # Generate mixed signals
        mixture_list = []
        for idx in range(min_samples):
            # Sum all source signals
            mixed_real = sum(self.signals[i][idx][:, 0] for i in range(self.num_sources))
            mixed_imag = sum(self.signals[i][idx][:, 1] for i in range(self.num_sources))
            mixed_signal = np.stack([mixed_real, mixed_imag], axis=1)
            mixture_list.append(mixed_signal)
        
        self.mixture = np.array(mixture_list)


def create_data_loaders(batch_size: int, data_choice: str, num_sources: int = 2,
                       train_ratio: float = 0.6, val_ratio: float = 0.2,
                       num_workers: int = 16, pin_memory: bool = True,
                       ) -> Tuple[DataLoader, DataLoader, Dict[float, DataLoader]]:
    """
    Unified interface for creating data loaders
    
    Args:
        batch_size: Batch size
        data_choice: Dataset selection
        num_sources: Number of source signals (2, 3, 4, ...)
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        num_workers: Number of data loading threads
        pin_memory: Whether to use pin_memory
    
    Returns:
        train_loader, val_loader, snr_loaders
    """
    
    # Select corresponding dataset class based on data type
    if data_choice in ['8PSK_M', '8PSK_M_8192', '8PSK_M_16384', '8PSK_M_32768', 'QPSK_16APSK', '8PSK_Rs', '16QAM_64QAM', '64QAM_64QAM', '64QAM_128QAM','16QAM_64QAM_128QAM']:
        dataset = _create_matlab_dataset(data_choice, num_sources)
    elif data_choice in ['2016', '2018', 'TorchSig']:
        dataset = _create_public_dataset(data_choice, num_sources)
    else:
        raise ValueError(f"Unsupported data type: {data_choice}")
    
    
    # Split dataset
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Group test data by SNR
    snr_to_indices = defaultdict(list)
    for idx in range(len(test_dataset)):
        _, _, snr = test_dataset[idx]
        snr_to_indices[snr].append(idx)
    
    snr_loaders = {}
    for snr, indices in snr_to_indices.items():
        snr_subset = torch.utils.data.Subset(test_dataset, indices)
        snr_loader = DataLoader(snr_subset, batch_size=batch_size, shuffle=False)
        snr_loaders[snr] = snr_loader
    
    # Create training and validation data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    print(f"Dataset statistics:")
    print(f"  Total samples: {dataset_size}")
    print(f"  Number of sources: {num_sources}")
    print(f"  Training: {train_size}, Validation: {val_size}, Test: {test_size}")
    print(f"  Input dimension: (batch_size, 2, signal_length)")
    print(f"  Output dimension: (batch_size, {2*num_sources}, signal_length)")
    
    return train_loader, val_loader, snr_loaders


def _create_matlab_dataset(data_choice: str, num_sources: int) -> MATLABSignalDataset:
    """Create MATLAB dataset"""
    
    # Define dataset configurations
    dataset_configs = {
        ("QAM", 2): {
            "base_path": "/root/autodl-tmp/QAM_M",
            "paths": [
                "/root/autodl-tmp/QAM_M",
                "/root/autodl-tmp/QAM_M_1"
            ],
            "snr_range": [30, 20],
            "file_range": range(1, 21),
            "signal_pattern": "16QAM_64QAM_Dataset_target_{i}_SNR={snr}dB.mat",
            "mixture_pattern": "16QAM_64QAM_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK", 3): {
            "base_path": "/root/autodl-tmp/dataset_imperfect",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "8PSK_Dataset_target_{i}_SNR={snr}dB.mat",
            "mixture_pattern": "8PSK_Dataset_mixed_{j}_SNR={snr}dB.mat"
        },
        ("8PSK_M", 2): {
            "base_path": "/root/autodl-tmp/IQUMamba/8PSK",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_M_8192", 2): {
            "base_path": "/root/autodl-tmp/dataset8192",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_M_16384", 2): {
            "base_path": "/root/autodl-tmp/dataset16384",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_M_32768", 2): {
            "base_path": "/root/autodl-tmp/dataset32768",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_M", 3): {
            "base_path": "/root/autodl-tmp/dataset_imperfect/8PSK",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "3Source_8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "3Source_8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("QPSK_16APSK", 2): {
            "base_path": "/root/autodl-tmp/IQUMamba/QPSK_16APSK",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/QPSK_16APSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/QPSK_16APSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_Rs", 2): {
            "base_path": "/root/autodl-tmp/IQUMamba/8PSK_Rs",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/2Source_8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/2Source_8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("8PSK_Rs", 3): {
            "base_path": "/root/autodl-tmp/IQUMamba/8PSK_Rs",
            "snr_range": range(-10, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "3Source_8PSK_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "3Source_8PSK_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("16QAM_64QAM", 2): {
            "base_path": "/root/autodl-tmp/IQUMamba/16QAM_64QAM",
            "snr_range": range(2, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "target/16QAM_64QAM_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "mixture/16QAM_64QAM_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("64QAM_64QAM", 2): {
            "base_path": "/root/autodl-tmp/IQUMamba/64QAM_64QAM",
            "snr_range": range(2, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "64QAM_64QAM_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "64QAM_64QAM_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("64QAM_128QAM", 2): {
            "base_path": "/root/autodl-tmp/IQUMamba/64QAM_128QAM",
            "snr_range": range(2, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "64QAM_128QAM_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "64QAM_128QAM_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },
        ("16QAM_64QAM_128QAM", 3): {
            "base_path": "/root/autodl-tmp/dataset_imperfect/16QAM_64QAM_128QAM",
            "snr_range": range(2, 31, 4),
            "file_range": range(1, 11),
            "signal_pattern": "16QAM_64QAM_128QAM_Dataset_target_{j}_SNR={snr}dB.mat",
            "mixture_pattern": "16QAM_64QAM_128QAM_Dataset_mixed_{i}_SNR={snr}dB.mat"
        },

    }
    
    config = dataset_configs.get((data_choice, num_sources))
    if not config:
        raise NotImplementedError(f'Unimplemented {data_choice} dataset configuration for {num_sources} sources')
    
    base_path = config["base_path"]
    
    # Handle special case: QAM 2-sources need multiple paths
    if (data_choice, num_sources) == ("QAM", 2):
        all_signal_paths = []
        all_mixture_paths = []
        for path in config["paths"]:
            signal_paths = [
                f'{path}/{config["signal_pattern"]}'.format(i=i, snr=snr)
                for snr in config["snr_range"] for i in config["file_range"]
            ]
            mixture_paths = [
                f'{path}/{config["mixture_pattern"]}'.format(i=i, snr=snr)
                for snr in config["snr_range"] for i in config["file_range"]
            ]
            all_signal_paths.extend(signal_paths)
            all_mixture_paths.extend(mixture_paths)
    else:
        # Use index variable j for signal_pattern, i for mixture_pattern
        all_signal_paths = [
            f'{base_path}/{config["signal_pattern"]}'.format(j=j, i=j, snr=snr)
            for snr in config["snr_range"] for j in config["file_range"]
        ]
        all_mixture_paths = [
            f'{base_path}/{config["mixture_pattern"]}'.format(i=i, j=i, snr=snr)
            for snr in config["snr_range"] for i in config["file_range"]
        ]
    
    return MATLABSignalDataset(all_signal_paths, all_mixture_paths, data_choice, num_sources)


def _create_public_dataset(data_choice: str, num_sources: int) -> PublicSignalDataset:
    """Create public dataset"""
    
    if data_choice == "2018":
        signal_paths = {
            'BPSK': [f'/root/IQUMamba1D/data/RML2018/BPSK/BPSK_SNR={snr}dB.mat'
                    for snr in range(-10, 31, 4)],
            'QPSK': [f'/root/IQUMamba1D/data/RML2018/QPSK/QPSK_SNR={snr}dB.mat'
                    for snr in range(-10, 31, 4)]
        }
        
    elif data_choice == "TorchSig":
        snr_list = [-10, -6, -2, 2, 6, 10, 14, 18, 22, 26, 30]
        signal_paths = {
            'BPSK': [f"/root/autodl-tmp/TorchSig/bpsk_SNR={snr}dB.mat"
                    for snr in snr_list],
            'QPSK': [f"/root/autodl-tmp/TorchSig/qpsk_SNR={snr}dB.mat"
                    for snr in snr_list]
        }
        
    elif data_choice == "2016":
        snr_list = list(range(-10, 19, 2))
        signal_paths = {
            'BPSK': [f"/root/autodl-tmp/RML2016/BPSK/MATBPSK_SNR={snr}dB.mat"
                    for snr in snr_list],
            'QPSK': [f"/root/autodl-tmp/RML2016/QPSK/MATQPSK_SNR={snr}dB.mat"
                    for snr in snr_list]
        }
        
        # For multi-source cases, add more modulation types
        if num_sources > 2:
            signal_paths.update({
                '8PSK': [f"/root/autodl-tmp/RML2016/MAT8PSK_SNR={snr}dB.mat"
                        for snr in snr_list],
                '16QAM': [f"/root/autodl-tmp/RML2016/MAT16QAM_SNR={snr}dB.mat"
                         for snr in snr_list]
            })
    
    else:
        raise NotImplementedError(f'Unimplemented public dataset type: {data_choice}')
    
    return PublicSignalDataset(signal_paths, data_choice, num_sources)


def print_dataset_info(train_loader, val_loader, snr_loaders):
    """Print dataset information"""
    print("\n=== Dataset Dimension Information ===")
    
    # Training set information
    print("\nTraining set:")
    for batch_idx, (input_signal, target, snr) in enumerate(train_loader):
        print(f"  Input signal shape: {input_signal.shape}")
        print(f"  Target signal shape: {target.shape}")
        print(f"  SNR examples: {snr[:5].tolist()}")
        break
    
    # Validation set information  
    print("\nValidation set:")
    for batch_idx, (input_signal, target, snr) in enumerate(val_loader):
        print(f"  Input signal shape: {input_signal.shape}")
        print(f"  Target signal shape: {target.shape}")
        break
    
    # Test set information (grouped by SNR)
    print("\nTest set (grouped by SNR):")
    for snr, loader in list(snr_loaders.items())[:3]:  # Only show first 3 SNRs
        print(f"  SNR = {snr} dB: {len(loader.dataset)} samples")


# Example usage
if __name__ == "__main__":
    # 2-source QAM dataset
    train_loader, val_loader, snr_loaders = create_data_loaders(
        batch_size=32, 
        data_choice="QAM", 
        num_sources=2
    )
    
    print_dataset_info(train_loader, val_loader, snr_loaders)