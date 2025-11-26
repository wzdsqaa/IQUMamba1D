import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from util.evaluation import test_model
from util.visualize import plot_losses
import os
import time
from collections import defaultdict


def train_model(
    model,
    scheduler, 
    train_loader, 
    val_loader, 
    snr_loaders, 
    criterion, 
    optimizer, 
    device, 
    num_epochs, 
    early_stop_patience, 
    logger, 
    results_folder, 
    data_choice,
    num_plots,
    batch_size,
    input_size,
    signal_names=None,
    # New parameters
    gradient_clip_norm=1.0,      # Gradient clipping
    save_checkpoint_every=10,    # Save checkpoint every N epochs
    accumulation_steps=1,        # Gradient accumulation steps
    warmup_epochs=5,             # Learning rate warmup
    use_mixed_precision=True,    # Mixed precision training
    log_interval=100,            # Logging interval
):
    # Ensure weights save directory exists
    weights_dir = f'/root/IQUMamba1D/results/{results_folder}/weights'
    os.makedirs(weights_dir, exist_ok=True)
    
    # Initialize best state
    best_val_loss = float('inf')
    no_improve_counter = 0
    best_model_weights = None
    
    # Record training history
    train_losses = []
    val_losses = []
    learning_rates = []
    epoch_times = []
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision and device.type == 'cuda' else None
    
    # Learning rate warmup scheduler
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # ========== Training Phase ==========
        model.train()
        train_loss = 0.0
        train_loss_components = defaultdict(float)  # Record loss components
        
        # Learning rate warmup
        current_lr = optimizer.param_groups[0]['lr']
        if epoch < warmup_epochs and warmup_epochs > 0:
            warmup_scheduler.step()
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', 
                 unit='batch', colour='green') as pbar:
            
            for batch_idx, (inputs, targets, snr) in enumerate(pbar):
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                snr = snr.to(device, non_blocking=True) if snr is not None else None
                
                # Mixed precision forward pass
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        if hasattr(criterion, 'forward') and 'snr' in criterion.forward.__code__.co_varnames:
                            total_loss = criterion(outputs, targets, snr) / accumulation_steps
                        else:
                            total_loss = criterion(outputs, targets) / accumulation_steps
                else:
                    outputs = model(inputs)
                    if hasattr(criterion, 'forward') and 'snr' in criterion.forward.__code__.co_varnames:
                        total_loss = criterion(outputs, targets, snr) / accumulation_steps
                    else:
                        total_loss = criterion(outputs, targets) / accumulation_steps
                
                # Backward pass
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    if gradient_clip_norm > 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    
                    # Optimizer step
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad()
                
                train_loss += total_loss.item() * accumulation_steps
                
                # Update progress bar
                if batch_idx % log_interval == 0:
                    pbar.set_postfix({
                        'Loss': f'{total_loss.item() * accumulation_steps:.4f}',
                        'LR': f'{current_lr:.2e}'
                    })
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        learning_rates.append(current_lr)
        
        # ========== Validation Phase ==========
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f'Validating Epoch {epoch+1}', 
                     unit='batch', colour='blue') as pbar:
                
                for inputs, targets, snr in pbar:
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    snr = snr.to(device, non_blocking=True) if snr is not None else None
                    
                    # Use mixed precision during validation too
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            outputs = model(inputs)
                            if hasattr(criterion, 'forward') and 'snr' in criterion.forward.__code__.co_varnames:
                                batch_loss = criterion(outputs, targets, snr)
                            else:
                                batch_loss = criterion(outputs, targets)
                    else:
                        outputs = model(inputs)
                        if hasattr(criterion, 'forward') and 'snr' in criterion.forward.__code__.co_varnames:
                            batch_loss = criterion(outputs, targets, snr)
                        else:
                            batch_loss = criterion(outputs, targets)
                    
                    val_loss += batch_loss.item()
                    pbar.set_postfix({'Val Loss': batch_loss.item()})
        
        # Learning rate scheduling (after warmup)
        if epoch >= warmup_epochs:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # ========== Model saving and early stopping logic ==========
        # Save checkpoint
        if (epoch + 1) % save_checkpoint_every == 0:
            checkpoint_path = os.path.join(weights_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            logger.info(f'Checkpoint saved: {checkpoint_path}')
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            improvement = best_val_loss - avg_val_loss
            logger.info(f'--> Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f} '
                       f'(improvement: {improvement:.4f})')
            best_val_loss = avg_val_loss
            no_improve_counter = 0
            
            # Save best model weights
            best_model_path = os.path.join(weights_dir, 'best_model_weights.pth')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f'Best model weights saved to {best_model_path}')
            best_model_weights = model.state_dict().copy()
        else:
            no_improve_counter += 1
            logger.info(f'No improvement for {no_improve_counter}/{early_stop_patience} epochs')
        
        # Detailed logging
        logger.info(f'Epoch [{epoch+1}/{num_epochs}] - '
                   f'Train Loss: {avg_train_loss:.4f}, '
                   f'Val Loss: {avg_val_loss:.4f}, '
                   f'LR: {current_lr:.2e}, '
                   f'Time: {epoch_time:.1f}s')
        
        # Check early stopping condition
        if no_improve_counter >= early_stop_patience:
            logger.info(f'Early stopping triggered after {epoch+1} epochs!')
            break
        
        # Memory cleanup
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # ========== Post-training processing ==========
    # Plot loss curves (including learning rate)
    plot_enhanced_losses(train_losses, val_losses, learning_rates, epoch_times, 
                        results_folder, signal_names=signal_names)
    # Plot loss curves
    plot_losses(train_losses, val_losses, results_folder, signal_names=signal_names)
    
    # Restore best model weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        logger.info("Loaded best model weights for final evaluation")
    
    # Final evaluation
    logger.info("Starting final model evaluation...")
    snr_metrics = test_model(
            model, snr_loaders, criterion, device, logger, results_folder,
            num_plots=1, num_points=256, input_size=input_size,
            data_choice=data_choice, signal_names=signal_names
        )
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'epoch_times': epoch_times,
        'best_val_loss': best_val_loss,
        'total_epochs': len(train_losses),
    }
    
    history_path = os.path.join(weights_dir, 'training_history.pth')
    torch.save(training_history, history_path)
    logger.info(f'Training history saved to {history_path}')
    
    # Training summary
    total_time = sum(epoch_times)
    avg_epoch_time = np.mean(epoch_times)
    logger.info(f'\n=== Training Summary ===')
    logger.info(f'Total training time: {total_time:.1f}s ({total_time/60:.1f}m)')
    logger.info(f'Average epoch time: {avg_epoch_time:.1f}s')
    logger.info(f'Best validation loss: {best_val_loss:.4f}')
    logger.info(f'Final learning rate: {optimizer.param_groups[0]["lr"]:.2e}')
    
    return training_history


def plot_enhanced_losses(train_losses, val_losses, learning_rates, epoch_times, 
                        results_folder, signal_names=None):
    """Plot enhanced loss curves, including learning rate and time information"""
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Learning rate curve
    ax2.plot(epochs, learning_rates, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Time per epoch
    ax3.plot(epochs, epoch_times, 'orange', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Training Time per Epoch')
    ax3.grid(True, alpha=0.3)
    
    # Cumulative time
    cumulative_time = np.cumsum(epoch_times) / 60  # Convert to minutes
    ax4.plot(epochs, cumulative_time, 'purple', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Cumulative Time (minutes)')
    ax4.set_title('Cumulative Training Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save image
    save_path = f'/root/IQUMamba1D/results/{results_folder}/enhanced_training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()