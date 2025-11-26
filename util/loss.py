import torch
import torch.nn.functional as F
import torch.nn as nn



def si_snr_loss(outputs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR) loss for two separated signals.
    
    Args:
        outputs: (B, 4, L) - [I1, Q1, I2, Q2]
        targets: (B, 4, L) - [I1, Q1, I2, Q2]
        eps: small value to avoid division by zero
    
    Returns:
        loss: scalar tensor (mean over batch and two signals)
    """
    B, C, L = outputs.shape
    assert C == 4, "Expected 4 channels: [I1, Q1, I2, Q2]"

    # Reshape to (B*2, L) for real and (B*2, L) for imag
    # Stack real and imag for both signals into (B*2, 2, L) then flatten to (B*2, 2*L)
    # But easier: treat each complex signal as 2D real vector of length 2*L

    # Extract signals
    out_I1 = outputs[:, 0, :]  # (B, L)
    out_Q1 = outputs[:, 1, :]
    out_I2 = outputs[:, 2, :]
    out_Q2 = outputs[:, 3, :]

    tar_I1 = targets[:, 0, :]
    tar_Q1 = targets[:, 1, :]
    tar_I2 = targets[:, 2, :]
    tar_Q2 = targets[:, 3, :]

    # Combine I and Q into (B, 2*L) real vectors
    out1 = torch.cat([out_I1, out_Q1], dim=1)  # (B, 2L)
    out2 = torch.cat([out_I2, out_Q2], dim=1)  # (B, 2L)
    tar1 = torch.cat([tar_I1, tar_Q1], dim=1)  # (B, 2L)
    tar2 = torch.cat([tar_I2, tar_Q2], dim=1)  # (B, 2L)

    # Compute SI-SNR for signal 1
    alpha1 = torch.sum(out1 * tar1, dim=1, keepdim=True) / (torch.sum(tar1 * tar1, dim=1, keepdim=True) + eps)
    target1_scaled = alpha1 * tar1
    noise1 = out1 - target1_scaled
    si_snr1 = torch.sum(target1_scaled ** 2, dim=1) / (torch.sum(noise1 ** 2, dim=1) + eps)
    si_snr1 = 10 * torch.log10(si_snr1 + eps)

    # Compute SI-SNR for signal 2
    alpha2 = torch.sum(out2 * tar2, dim=1, keepdim=True) / (torch.sum(tar2 * tar2, dim=1, keepdim=True) + eps)
    target2_scaled = alpha2 * tar2
    noise2 = out2 - target2_scaled
    si_snr2 = torch.sum(target2_scaled ** 2, dim=1) / (torch.sum(noise2 ** 2, dim=1) + eps)
    si_snr2 = 10 * torch.log10(si_snr2 + eps)

    # Average over batch and two signals, then negate for loss
    si_snr_total = (si_snr1 + si_snr2) / 2.0  # (B,)
    loss = -torch.mean(si_snr_total)

    return loss


def si_snr_mse_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Combined SI-SNR + MSE loss for blind source separation of two complex signals.
    
    Args:
        outputs: (B, 4, L) - [I1, Q1, I2, Q2]
        targets: (B, 4, L) - [I1, Q1, I2, Q2]
        alpha: weight for SI-SNR loss (default: 1.0)
        beta: weight for MSE loss (default: 0.01, since MSE is usually larger in magnitude)
        eps: small constant for numerical stability
    
    Returns:
        loss: scalar tensor = alpha * (-SI-SNR) + beta * MSE
    """
    B, C, L = outputs.shape
    assert C == 4, "Expected 4 channels: [I1, Q1, I2, Q2]"

    # ========== 1. Compute SI-SNR loss (as before) ==========
    out_I1, out_Q1 = outputs[:, 0, :], outputs[:, 1, :]
    out_I2, out_Q2 = outputs[:, 2, :], outputs[:, 3, :]
    tar_I1, tar_Q1 = targets[:, 0, :], targets[:, 1, :]
    tar_I2, tar_Q2 = targets[:, 2, :], targets[:, 3, :]

    # Flatten to (B, 2L)
    out1 = torch.cat([out_I1, out_Q1], dim=1)
    out2 = torch.cat([out_I2, out_Q2], dim=1)
    tar1 = torch.cat([tar_I1, tar_Q1], dim=1)
    tar2 = torch.cat([tar_I2, tar_Q2], dim=1)

    # SI-SNR for signal 1
    alpha1 = torch.sum(out1 * tar1, dim=1, keepdim=True) / (torch.sum(tar1 * tar1, dim=1, keepdim=True) + eps)
    target1_scaled = alpha1 * tar1
    noise1 = out1 - target1_scaled
    si_snr1 = torch.sum(target1_scaled ** 2, dim=1) / (torch.sum(noise1 ** 2, dim=1) + eps)
    si_snr1 = 10 * torch.log10(si_snr1 + eps)

    # SI-SNR for signal 2
    alpha2 = torch.sum(out2 * tar2, dim=1, keepdim=True) / (torch.sum(tar2 * tar2, dim=1, keepdim=True) + eps)
    target2_scaled = alpha2 * tar2
    noise2 = out2 - target2_scaled
    si_snr2 = torch.sum(target2_scaled ** 2, dim=1) / (torch.sum(noise2 ** 2, dim=1) + eps)
    si_snr2 = 10 * torch.log10(si_snr2 + eps)

    si_snr_mean = (si_snr1 + si_snr2) / 2.0
    si_snr_loss_val = -torch.mean(si_snr_mean)  # higher SI-SNR → lower loss

    # ========== 2. Compute MSE loss ==========
    mse_loss_val = nn.MSELoss()(outputs, targets)

    # ========== 3. Combine ==========
    total_loss = alpha * si_snr_loss_val + beta * mse_loss_val

    return total_loss

