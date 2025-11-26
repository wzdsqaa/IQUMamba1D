import torch

def l2_norm(s1, s2):
    """
    Calculate the inner product of two signals
    Supports dimensions: (B, C, L) where B=batch_size, C=channels, L=length
    """
    # Sum over channel and length dimensions, keeping batch dimension
    norm = torch.sum(s1 * s2, dim=(-2, -1), keepdim=True)  # (B, 1, 1)
    return norm

def calculate_si_sdr(s_hat, s_target, eps=1e-8):
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    
    Args:
        s_hat: Separated signal (B, C, L) where C=2*N, N is number of signals
        s_target: Label signal/target signal (B, C, L)
        eps: Small constant to prevent division by zero
    
    Returns:
        SI-SDR value (B,) or scalar
    """
    # Calculate optimal scaling factor α = <s_hat, s_target> / ||s_target||^2
    s_hat_target_norm = l2_norm(s_hat, s_target)  # (B, 1, 1)
    target_target_norm = l2_norm(s_target, s_target)  # (B, 1, 1)
    alpha = s_hat_target_norm / (target_target_norm + eps)  # (B, 1, 1)
    
    # Calculate scaled target signal
    e_target = alpha * s_target  # (B, C, L)
    
    # Calculate residual
    e_res = s_hat - e_target  # (B, C, L)
    
    # Calculate SI-SDR
    target_power = l2_norm(e_target, e_target)  # (B, 1, 1)
    residual_power = l2_norm(e_res, e_res)  # (B, 1, 1)
    
    si_sdr = 10 * torch.log10((target_power) / (residual_power + eps) + eps)  # (B, 1, 1)
    si_sdr = si_sdr.squeeze(-1).squeeze(-1)  # (B,)
    
    return si_sdr.mean() if si_sdr.numel() > 1 else si_sdr.item()

def sum_sources_to_mixture_format(source_signals):
    """
    Sum all source signals to mixture signal format
    From [S1(B,C,L), S2(B,C,L), ...] to (B, 2, L)
    
    Args:
        source_signals: List of source signals, each (B, C, L), where C=2*N
    
    Returns:
        Sum in mixture signal format (B, 2, L)
    """
    if not source_signals:
        return None
    
    # Get dimension information
    B, C, L = source_signals[0].shape
    num_signals_per_source = C // 2  # Number of signals per source
    
    # Initialize total mixture signal
    total_mixture = torch.zeros(B, 2, L, device=source_signals[0].device, dtype=source_signals[0].dtype)
    
    # For each source signal, sum the real and imaginary parts of all its sub-signals
    for source in source_signals:
        for i in range(num_signals_per_source):
            # Add real and imaginary parts of the i-th signal to the mixture
            total_mixture[:, 0, :] += source[:, 2*i, :]     # Real part
            total_mixture[:, 1, :] += source[:, 2*i+1, :]   # Imaginary part
    
    return total_mixture

def calculate_interference_signal(mixture, source_signals):
    """
    Calculate interference signal = mixture signal - sum of all source signals
    
    Args:
        mixture: Mixture signal (B, 2, L)
        source_signals: List of source signals, each (B, C, L)
    
    Returns:
        Interference signal (B, 2, L)
    """
    # Sum all source signals to mixture format
    total_sources_mixture = sum_sources_to_mixture_format(source_signals)
    
    if total_sources_mixture is None:
        raise ValueError("No source signals provided")
    
    # Calculate interference signal: mixture - sum of all source signals
    interference_mixture = mixture - total_sources_mixture  # (B, 2, L)
    
    return interference_mixture

def expand_interference_to_target_format(interference_mixture, target_channels):
    """
    Expand interference signal from mixture format (B, 2, L) to target format (B, C, L)
    Assume interference affects all signals equally
    
    Args:
        interference_mixture: Interference signal (B, 2, L)
        target_channels: Target channel number C=2*N
    
    Returns:
        Expanded interference signal (B, C, L)
    """
    B, _, L = interference_mixture.shape
    num_signals = target_channels // 2
    
    # Create expanded interference signal
    expanded_interference = torch.zeros(B, target_channels, L, 
                                      device=interference_mixture.device, 
                                      dtype=interference_mixture.dtype)
    
    # For each signal position, copy real and imaginary parts of interference
    for i in range(num_signals):
        expanded_interference[:, 2*i, :] = interference_mixture[:, 0, :]     # Real part
        expanded_interference[:, 2*i+1, :] = interference_mixture[:, 1, :]   # Imaginary part
    
    return expanded_interference

def calculate_si_sir_multisource(s_hat, s_target, mixture, all_source_signals, eps=1e-8):
    """
    Calculate Scale-Invariant Signal-to-Interference Ratio (SI-SIR) in multi-source separation scenario
    
    Args:
        s_hat: Separated signal (B, C, L)
        s_target: Current target signal (B, C, L)
        mixture: Mixture signal (B, 2, L)
        all_source_signals: List of all source signals, each (B, C, L)
        eps: Small constant to prevent division by zero
    
    Returns:
        SI-SIR value
    """
    # Calculate optimal scaling factor
    s_hat_target_norm = l2_norm(s_hat, s_target)
    target_target_norm = l2_norm(s_target, s_target)
    alpha = s_hat_target_norm / (target_target_norm + eps)
    
    # Calculate scaled target signal
    e_target = alpha * s_target
    
    # Calculate residual
    e_res = s_hat - e_target
    
    # Calculate interference signal (B, 2, L)
    interference_mixture = calculate_interference_signal(mixture, all_source_signals)
    
    # Expand interference signal to same format as target signal (B, C, L)
    target_channels = s_target.shape[1]
    interference = expand_interference_to_target_format(interference_mixture, target_channels)
    
    # Calculate interference part: projection of e_res on interference signal
    e_res_interf_norm = l2_norm(e_res, interference)
    interf_interf_norm = l2_norm(interference, interference)
    beta = e_res_interf_norm / (interf_interf_norm + eps)
    e_interf = beta * interference
    
    # Calculate SI-SIR
    target_power = l2_norm(e_target, e_target)
    interf_power = l2_norm(e_interf, e_interf)
    
    si_sir = 10 * torch.log10((target_power) / (interf_power + eps) + eps)
    si_sir = si_sir.squeeze(-1).squeeze(-1)
    
    return si_sir.mean() if si_sir.numel() > 1 else si_sir.item()

def calculate_si_sar_multisource(s_hat, s_target, mixture, all_source_signals, eps=1e-8):
    """
    Calculate Scale-Invariant Signal-to-Artifacts Ratio (SI-SAR) in multi-source separation scenario
    
    Args:
        s_hat: Separated signal (B, C, L)
        s_target: Current target signal (B, C, L)
        mixture: Mixture signal (B, 2, L)
        all_source_signals: List of all source signals, each (B, C, L)
        eps: Small constant to prevent division by zero
    
    Returns:
        SI-SAR value
    """
    # Calculate optimal scaling factor
    s_hat_target_norm = l2_norm(s_hat, s_target)
    target_target_norm = l2_norm(s_target, s_target)
    alpha = s_hat_target_norm / (target_target_norm + eps)
    
    # Calculate scaled target signal
    e_target = alpha * s_target
    
    # Calculate residual
    e_res = s_hat - e_target
    
    # Calculate interference signal (B, 2, L)
    interference_mixture = calculate_interference_signal(mixture, all_source_signals)
    
    # Expand interference signal to same format as target signal (B, C, L)
    target_channels = s_target.shape[1]
    interference = expand_interference_to_target_format(interference_mixture, target_channels)
    
    # Calculate interference part
    e_res_interf_norm = l2_norm(e_res, interference)
    interf_interf_norm = l2_norm(interference, interference)
    beta = e_res_interf_norm / (interf_interf_norm + eps)
    e_interf = beta * interference
    
    # Calculate artifact part: e_artif = e_res - e_interf
    e_artif = e_res - e_interf
    
    # Calculate SI-SAR
    target_power = l2_norm(e_target, e_target)
    artif_power = l2_norm(e_artif, e_artif)
    
    si_sar = 10 * torch.log10((target_power) / (artif_power + eps) + eps)
    si_sar = si_sar.squeeze(-1).squeeze(-1)
    
    return si_sar.mean() if si_sar.numel() > 1 else si_sar.item()


def calculate_correlation(prediction, target):

    pred_flat = prediction.flatten(start_dim=1)  # (B, C*L)
    target_flat = target.flatten(start_dim=1)    # (B, C*L)
    
    # Calculate correlation coefficient for each batch
    correlations = []
    for i in range(pred_flat.shape[0]):
        corr_matrix = torch.corrcoef(torch.stack([pred_flat[i], target_flat[i]]))
        corr = corr_matrix[0, 1] if not torch.isnan(corr_matrix[0, 1]) else 0
        correlations.append(corr)
    
    return torch.tensor(correlations).mean()