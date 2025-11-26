import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.visualize import plot_signals, plot_correlation_vs_snr_enhanced
from util.SI_Metrics import calculate_si_sdr, calculate_si_sir_multisource, calculate_si_sar_multisource


def calculate_correlation(prediction, target):
    """Calculate overall correlation coefficient"""
    pred_flat = prediction.flatten()
    target_flat = target.flatten()
    corr_matrix = torch.corrcoef(torch.stack([pred_flat, target_flat]))
    return corr_matrix[0, 1] if not torch.isnan(corr_matrix[0, 1]) else 0


def calculate_correlation_per_source(prediction, target, num_sources):
    """
    Calculate correlation coefficient for each source
    
    Args:
        prediction: Prediction output (B, C, L), C = num_sources * 2
        target: Target signal (B, C, L), C = num_sources * 2
        num_sources: Number of sources
    
    Returns:
        list: Correlation coefficients for each source
    """
    correlations = []
    
    for i in range(num_sources):
        # Each source contains real and imaginary parts in two channels
        real_idx = i * 2
        imag_idx = i * 2 + 1
        
        # Extract prediction and target for current source
        pred_source = prediction[:, [real_idx, imag_idx], :].flatten()
        target_source = target[:, [real_idx, imag_idx], :].flatten()
        
        # Calculate correlation coefficient
        if len(pred_source) > 0 and len(target_source) > 0:
            corr_matrix = torch.corrcoef(torch.stack([pred_source, target_source]))
            corr = corr_matrix[0, 1] if not torch.isnan(corr_matrix[0, 1]) else 0
            correlations.append(corr)
        else:
            correlations.append(torch.tensor(0.0))
    
    return correlations


def calculate_metrics_per_source(prediction, target, inputs, num_sources):
    """
    Calculate metrics for each source
    
    Args:
        prediction: Prediction output (B, C, L)
        target: Target signal (B, C, L)
        inputs: Input mixed signal (B, 2, L)
        num_sources: Number of sources
    
    Returns:
        dict: Metrics dictionary for each source
    """
    source_metrics = {}
    
    for i in range(num_sources):
        # Each source contains real and imaginary parts in two channels
        real_idx = i * 2
        imag_idx = i * 2 + 1
        
        # Extract prediction and target for current source
        pred_source = prediction[:, [real_idx, imag_idx], :]  # (B, 2, L)
        target_source = target[:, [real_idx, imag_idx], :]    # (B, 2, L)
        
        # Calculate metrics
        try:
            sdr = calculate_si_sdr(pred_source, target_source)
            
            # For single source SIR and SAR calculation, need to construct list of all sources
            all_sources = []
            for j in range(num_sources):
                r_idx = j * 2
                i_idx = j * 2 + 1
                source_j = target[:, [r_idx, i_idx], :]
                all_sources.append(source_j)
            
            sir = calculate_si_sir_multisource(pred_source, target_source, 
                                             mixture=inputs, all_source_signals=all_sources)
            sar = calculate_si_sar_multisource(pred_source, target_source, 
                                             mixture=inputs, all_source_signals=all_sources)
            
            # Calculate correlation coefficient
            corr = calculate_correlation(pred_source, target_source)
            
        except Exception as e:
            print(f"Error calculating metrics for source {i}: {e}")
            sdr = torch.tensor(0.0)
            sir = torch.tensor(0.0)
            sar = torch.tensor(0.0)
            corr = torch.tensor(0.0)
        
        source_metrics[i] = {
            'SI-SDR': sdr,
            'SI-SIR': sir,
            'SI-SAR': sar,
            'Correlation': corr
        }
    
    return source_metrics


def split_tensor_by_channel(x: torch.Tensor, N: int):
    """
    x   : Tensor of shape (B, C, L)
    N   : Expected number of splits (must be even, and N <= C)
    return : list of N tensors, each (B, 2, L)
    """
    B, C, L = x.shape
    assert N <= C, "N must be <= number of channels C"
    
    # Split channel dimension into (N, 2) then transpose
    y = x.view(B, N, 2, L).transpose(1, 2)          # (B, 2, N, L)
    return [t.squeeze(2) for t in y.split(1, dim=2)]


def test_model(model, snr_loaders, criterion, device, logger, results_folder, 
               num_plots=1, num_points=1024, input_size=1024, data_choice='2018',
               signal_names=None):
    """
    Enhanced test_model function that supports independent evaluation for each source
    """
    model.eval()
    
    num_source = len(signal_names) if signal_names else 2
    snr_metrics = {}
    
    # Store metrics for each source for CSV
    all_metrics_list = []
    
    # Store correlation coefficients for each source for plotting
    source_correlations = {i: [] for i in range(num_source)}
    overall_correlations = []
    snr_list = []
    
    
    with torch.no_grad():
        # Global warmup
        for _ in range(3):
            dummy_input = torch.randn(1, 2, input_size).to(device)
            _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    total_time = 0.0
    total_samples = 0
    
    for snr, loader in sorted(snr_loaders.items(), key=lambda x: x[0]):
        test_loss = 0.0
        test_corr = 0.0
        test_sdr = 0.0
        test_sir = 0.0
        test_sar = 0.0
        
        # Cumulative metrics for each source
        source_metrics_sum = {i: {'SI-SDR': 0.0, 'SI-SIR': 0.0, 'SI-SAR': 0.0, 'Correlation': 0.0} 
                             for i in range(num_source)}
        
        visualization_count = 0
        visualization_done = False
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, _) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                

                start_time = time.perf_counter()
                outputs = model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                if batch_idx >= 0:
                    total_time += (end_time - start_time)
                    total_samples += inputs.size(0)
                
                # Calculate overall metrics
                test_loss += criterion(outputs, targets).item()
                test_corr += calculate_correlation(outputs, targets)
                test_sdr += calculate_si_sdr(outputs, targets)
                test_sir += calculate_si_sir_multisource(outputs, targets, mixture=inputs, 
                                                       all_source_signals=split_tensor_by_channel(x=targets, N=num_source))
                test_sar += calculate_si_sar_multisource(outputs, targets, mixture=inputs, 
                                                       all_source_signals=split_tensor_by_channel(x=targets, N=num_source))
                
                # Calculate metrics for each source
                source_metrics_batch = calculate_metrics_per_source(outputs, targets, inputs, num_source)
                
                # Accumulate metrics for each source
                for source_idx in range(num_source):
                    for metric_name in ['SI-SDR', 'SI-SIR', 'SI-SAR', 'Correlation']:
                        source_metrics_sum[source_idx][metric_name] += source_metrics_batch[source_idx][metric_name]
                
                # Plot signal comparison
                if batch_idx == 0 and num_plots > 0:
                    for sample_idx in range(min(num_plots, inputs.size(0))):
                        input_sample = inputs[sample_idx].cpu().numpy()
                        target_sample = targets[sample_idx].cpu().numpy()
                        output_sample = outputs[sample_idx].cpu().numpy()
                        plot_signals(input_sample, target_sample, output_sample, 
                                   sample_idx, snr, logger, results_folder, num_points, signal_names=signal_names)

        # Calculate average metrics
        num_batches = len(loader)
        avg_loss = test_loss / num_batches
        avg_sdr = test_sdr / num_batches
        avg_corr = test_corr / num_batches
        avg_sir = test_sir / num_batches
        avg_sar = test_sar / num_batches
        
        # Calculate average metrics for each source
        source_avg_metrics = {}
        for source_idx in range(num_source):
            source_avg_metrics[source_idx] = {}
            for metric_name in ['SI-SDR', 'SI-SIR', 'SI-SAR', 'Correlation']:
                source_avg_metrics[source_idx][metric_name] = source_metrics_sum[source_idx][metric_name] / num_batches
        
        # Store overall metrics
        snr_metrics[snr] = {
            'Loss': avg_loss,
            'Correlation': avg_corr,
            'SI-SDR': avg_sdr,
            'SI-SIR': avg_sir,
            'SI-SAR': avg_sar,
            'Source_Metrics': source_avg_metrics
        }
        
        # Store correlation coefficients for plotting
        snr_list.append(snr)
        overall_correlations.append(avg_corr.item())
        for source_idx in range(num_source):
            source_correlations[source_idx].append(source_avg_metrics[source_idx]['Correlation'].item())
        
        # Add overall metrics for CSV
        metrics_row = {
            'SNR': snr,
            'Source': 'Overall',
            'Signal_Type': 'All_Sources',
            'Loss': avg_loss,
            'Correlation': avg_corr.item(),
            'SI-SDR': avg_sdr.item(),
            'SI-SIR': avg_sir.item(),
            'SI-SAR': avg_sar.item()
        }
        all_metrics_list.append(metrics_row)
        
        # Add metrics for each source for CSV
        for source_idx in range(num_source):
            signal_name = signal_names[source_idx] if signal_names else f'Source_{source_idx}'
            source_row = {
                'SNR': snr,
                'Source': f'Source_{source_idx}',
                'Signal_Type': signal_name,
                'Loss': avg_loss,  # Loss is overall
                'Correlation': source_avg_metrics[source_idx]['Correlation'].item(),
                'SI-SDR': source_avg_metrics[source_idx]['SI-SDR'].item(),
                'SI-SIR': source_avg_metrics[source_idx]['SI-SIR'].item(),
                'SI-SAR': source_avg_metrics[source_idx]['SI-SAR'].item()
            }
            all_metrics_list.append(source_row)
        
        # Log results
        logger.info(f'SNR {snr}dB:')
        logger.info(f'\tOverall - Loss: {avg_loss:.8f}, Correlation: {avg_corr:.8f}')
        logger.info(f'\tOverall - SI-SDR: {avg_sdr:.8f} dB, SI-SIR: {avg_sir:.8f} dB, SI-SAR: {avg_sar:.8f} dB')
        
        for source_idx in range(num_source):
            signal_name = signal_names[source_idx] if signal_names else f'Source_{source_idx}'
            metrics = source_avg_metrics[source_idx]
            logger.info(f'\t{signal_name} - Correlation: {metrics["Correlation"]:.8f}')
            logger.info(f'\t{signal_name} - SI-SDR: {metrics["SI-SDR"]:.8f} dB, SI-SIR: {metrics["SI-SIR"]:.8f} dB, SI-SAR: {metrics["SI-SAR"]:.8f} dB')
    
    # Save detailed CSV file
    df = pd.DataFrame(all_metrics_list)
    csv_path = f"/root/IQUMamba1D/results/{results_folder}/detailed_metrics_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Detailed metrics saved to {csv_path}")
    
    # Plot enhanced correlation vs SNR
    plot_correlation_vs_snr_enhanced(snr_list, overall_correlations, source_correlations, 
                                   results_folder, signal_names)
    
    # Output average time
    if total_samples > 0:
        avg_time_per_sample = (total_time / total_samples) * 1000
        logger.info(f"Global Avg Inference Time per Sample: {avg_time_per_sample:.4f} ms")
    
    return snr_metrics