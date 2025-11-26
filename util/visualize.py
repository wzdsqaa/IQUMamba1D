from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np

from datetime import datetime

import matplotlib.font_manager as fm

# Set Euclid font (please adjust the actual font file path)
def setup_euclid_font():
    # Try to set Euclid font
    try:
        # Try to add Euclid font
        fm.fontManager.addfont('/root/fonts/euclid.ttf')  # Replace with actual path
        plt.rcParams['font.family'] = 'Euclid'
    except:
        # If Euclid font is not available, use default font
        print("Euclid font not available, using default font")
        pass

# Call font setup before class definitions
setup_euclid_font()


def plot_losses(train_losses, val_losses, results_folder, signal_names=['BPSK', 'QPSK']):
    """
    Plot the loss values on training and validation sets as a function of training epochs.

    Parameters:
    - train_losses: List of loss values on training set
    - val_losses: List of loss values on validation set
    - results_folder: Folder path to save the image
    - signal_names: List of signal source names, e.g., ['BPSK', 'QPSK'] or ['BPSK', 'QPSK', '8PSK']
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.style.use('seaborn-v0_8')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times', 'DejaVu Serif', 'serif']
    plt.rcParams['font.size'] = 12
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(epochs, train_losses, label='Training Loss', 
             color='#2E86AB', linewidth=2.5, alpha=0.8)
    plt.plot(epochs, val_losses, label='Validation Loss', 
             color='#A23B72', linewidth=2.5, alpha=0.8)
    
    signal_info = ' + '.join(signal_names)
    plt.title(f'Training and Validation Loss\n({signal_info} Blind Source Separation)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epochs', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')

    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)

    plt.grid(True, alpha=0.3, linestyle='--')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()

    plt.savefig(f'/root/IQUMamba1D/results/{results_folder}/loss_plot.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_signals(input_signal, target, prediction, sample_idx, snr, logger, results_folder, 
                num_points=512, signal_names=['BPSK', 'QPSK']):
    """
    Plot comparison of input signal, target signal, and predicted signal.
    Create a separate plot for each source, including comparison between target and predicted signals.

    Parameters:
    - input_signal: Input mixed signal
    - target: Target signal
    - prediction: Predicted signal
    - sample_idx: Sample index
    - snr: Signal-to-noise ratio
    - logger: Logger
    - results_folder: Results save folder
    - signal_names: List of signal source names, e.g., ['BPSK', 'QPSK'] or ['BPSK', 'QPSK', '8PSK']
    - num_points: Number of sampling points to display
    """
    

    plt.rcParams.update(plt.rcParamsDefault)  # Reset to default settings
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['font.family'] = 'Euclid'
    #plt.rcParams['font.serif'] = ['Times', 'DejaVu Serif', 'serif']
    plt.rcParams['font.size'] = 30
    
    # Dynamically determine number of signal sources
    num_sources = len(signal_names)
    
    # Create color mapping
    target_colors = ["#2E8B57", '#FF6347']  # Target signal colors: sea green for real, tomato red for imaginary
    pred_colors = ['#1E90FF', '#FF8C00']    # Predicted signal colors: dodger blue for real, dark orange for imaginary
    
    # Generate timestamped filename prefix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    signal_info = "_".join(signal_names)

    
    # 2. Create separation result comparison plots for each source
    for i, signal_name in enumerate(signal_names):
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), facecolor='white')
        
        # Set subplot background to white
        for ax in axes:
            ax.set_facecolor('white')
        
        # Calculate real and imaginary indices for current source
        real_idx = i * 2
        imag_idx = i * 2 + 1
        
        # Plot target signal
        axes[0].plot(target[real_idx, :num_points], 
                    label=f'{signal_name} Real (Target)', 
                    color=target_colors[0], linewidth=2)
        axes[0].plot(target[imag_idx, :num_points], 
                    label=f'{signal_name} Imag (Target)', 
                    color=target_colors[1], linewidth=2, linestyle='-')
        
        #axes[0].set_title(f'Target {signal_name} Signal', fontsize=30)
        axes[0].legend(loc='upper right', frameon=True, fancybox=False, fontsize=20)
        axes[0].grid(True, alpha=0.3, linestyle='-', color='lightgray')
        axes[0].set_ylabel('Amplitude', fontsize=30)
        
        # Plot predicted signal
        axes[1].plot(prediction[real_idx, :num_points], 
                    label=f'{signal_name} Real (Predicted)', 
                    color=pred_colors[0], linewidth=2)
        axes[1].plot(prediction[imag_idx, :num_points], 
                    label=f'{signal_name} Imag (Predicted)', 
                    color=pred_colors[1], linewidth=2, linestyle='-')
        
        #axes[1].set_title(f'Predicted {signal_name} Signal', fontsize=30)
        axes[1].legend(loc='upper right', frameon=True, fancybox=False, fontsize=20)
        axes[1].grid(True, alpha=0.3, linestyle='-', color='lightgray')
        axes[1].set_ylabel('Amplitude', fontsize=30)
        axes[1].set_xlabel('Sample Index', fontsize=30)
        
        # Add overall title
        #fig.suptitle(f'{signal_name} Signal Separation Results (SNR={snr}dB)', fontsize=30, y=0.95)
        
        plt.tight_layout()
        
        # Save separation result plot for each source
        source_filename = f'/root/IQUMamba1D/results/{results_folder}/saved_plots/{signal_name}_separation_SNR_{snr}dB_sample_{sample_idx}_{timestamp}.png'
        plt.savefig(source_filename, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f'Saved {signal_name} separation plot to {source_filename}')
    
    # 3. Optional: Create an overview plot showing prediction comparison for all sources
    fig_overview, axes_overview = plt.subplots(num_sources, 1, figsize=(12, 4*num_sources), facecolor='white')
    
    # If only one source, ensure axes_overview is a list
    if num_sources == 1:
        axes_overview = [axes_overview]
    
    # Set background of all subplots to white
    for ax in axes_overview:
        ax.set_facecolor('white')
    
    for i, signal_name in enumerate(signal_names):
        real_idx = i * 2
        imag_idx = i * 2 + 1
        
        # Plot both target and predicted signals in the same subplot
        axes_overview[i].plot(target[real_idx, :num_points], 
                             label=f'Target Real', 
                             color=target_colors[0], linewidth=2)
        axes_overview[i].plot(target[imag_idx, :num_points], 
                             label=f'Target Imag', 
                             color=target_colors[1], linewidth=2, linestyle='--')
        
        axes_overview[i].plot(prediction[real_idx, :num_points], 
                             label=f'Pred Real', 
                             color=pred_colors[0], linewidth=1.5)
        axes_overview[i].plot(prediction[imag_idx, :num_points], 
                             label=f'Pred Imag', 
                             color=pred_colors[1], linewidth=1.5, linestyle=':')
        
        axes_overview[i].set_title(f'{signal_name} Signal: Target vs Prediction', 
                                  fontsize=12)
        axes_overview[i].legend(loc='upper right', frameon=True, fancybox=False, fontsize=30)
        axes_overview[i].grid(True, alpha=0.3, linestyle='-', color='lightgray')
        axes_overview[i].set_ylabel('Amplitude', fontsize=30)
    
    # Add x-axis label only to the last subplot
    axes_overview[-1].set_xlabel('Sample Index', fontsize=30)
    
    # Add overall title
    fig_overview.suptitle(f'Blind Source Separation Overview (SNR={snr}dB)\nSignals: {" + ".join(signal_names)}', 
                         fontsize=15, y=0.98)
    
    plt.tight_layout()
    
    # Save overview plot
    overview_filename = f'/root/IQUMamba1D/results/{results_folder}/saved_plots/separation_overview_{signal_info}_SNR_{snr}dB_sample_{sample_idx}_{timestamp}.png'
    plt.savefig(overview_filename, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f'Saved separation overview plot to {overview_filename}')
    
    logger.info(f'All plots saved successfully for {len(signal_names)} sources')

def plot_correlation_vs_snr_enhanced(snr_list, overall_correlations, source_correlations, 
                                    results_folder, signal_names=None):
    """
    Plot enhanced correlation vs SNR graph, including correlations for each source and overall correlation
    
    Args:
        snr_list: SNR list
        overall_correlations: Overall correlation list
        source_correlations: Dictionary of correlations for each source {source_idx: [correlations]}
        results_folder: Results save folder
        signal_names: List of signal source names
    """
    # Set matplotlib style
    plt.style.use('seaborn-v0_8')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times', 'DejaVu Serif', 'serif']
    plt.rcParams['font.size'] = 12
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Color mapping
    colors = plt.cm.Set1(np.linspace(0, 1, len(source_correlations)))
    
    # Subplot 1: All curves together
    # Plot correlations for each source
    for source_idx, correlations in source_correlations.items():
        signal_name = signal_names[source_idx] if signal_names else f'Source {source_idx}'
        ax1.plot(snr_list, correlations, marker='s', linestyle='--', 
                color=colors[source_idx], linewidth=2, markersize=6, 
                alpha=0.8, label=f'{signal_name}')
    
    # Plot overall correlation
    ax1.plot(snr_list, overall_correlations, marker='o', linestyle='-', 
            color='#2E86AB', linewidth=3, markersize=8, 
            markerfacecolor='#A23B72', markeredgecolor='white', 
            markeredgewidth=2, alpha=0.9, label='Overall')
    
    # Add signal source info to title
    signal_info = ' + '.join(signal_names) if signal_names else f'{len(source_correlations)} Sources'
    ax1.set_title(f'Correlation vs SNR - All Sources\n({signal_info} BSS)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Correlation', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1.05)
    
    # Beautify axes
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    
    # Subplot 2: Each source displayed separately (stacked subplots)
    num_sources = len(source_correlations)
    fig2, axes2 = plt.subplots(num_sources + 1, 1, figsize=(12, 4 * (num_sources + 1)))
    
    if num_sources == 0:  # Ensure axes2 is a list
        axes2 = [axes2]
    elif num_sources == 1:
        axes2 = [axes2[0], axes2[1]]
    
    # Plot each source
    for source_idx, correlations in source_correlations.items():
        signal_name = signal_names[source_idx] if signal_names else f'Source {source_idx}'
        
        axes2[source_idx].plot(snr_list, correlations, marker='s', linestyle='-', 
                              color=colors[source_idx], linewidth=2.5, markersize=7, 
                              markerfacecolor=colors[source_idx], markeredgecolor='white', 
                              markeredgewidth=1.5, alpha=0.9)
        
        # Add value labels
        for snr, corr in zip(snr_list, correlations):
            if len(snr_list) <= 10:  # Only show labels when not too many SNR points
                axes2[source_idx].annotate(f'{corr:.3f}', (snr, corr), 
                                         textcoords="offset points", xytext=(0,8), ha='center',
                                         fontsize=9, fontweight='bold')
        
        axes2[source_idx].set_title(f'{signal_name} Correlation vs SNR', 
                                   fontsize=13, fontweight='bold')
        axes2[source_idx].set_ylabel('Correlation', fontsize=11, fontweight='bold')
        axes2[source_idx].grid(True, alpha=0.3, linestyle='--')
        axes2[source_idx].set_ylim(0, 1.05)
        
        # Beautify axes
        axes2[source_idx].spines['top'].set_visible(False)
        axes2[source_idx].spines['right'].set_visible(False)
        axes2[source_idx].spines['left'].set_linewidth(1.5)
        axes2[source_idx].spines['bottom'].set_linewidth(1.5)
    
    # Plot overall correlation
    axes2[-1].plot(snr_list, overall_correlations, marker='o', linestyle='-', 
                  color='#2E86AB', linewidth=3, markersize=8, 
                  markerfacecolor='#A23B72', markeredgecolor='white', 
                  markeredgewidth=2, alpha=0.9)
    
    # Add value labels for overall correlation
    for snr, corr in zip(snr_list, overall_correlations):
        if len(snr_list) <= 10:
            axes2[-1].annotate(f'{corr:.3f}', (snr, corr), 
                             textcoords="offset points", xytext=(0,10), ha='center',
                             fontsize=10, fontweight='bold')
    
    axes2[-1].set_title('Overall Correlation vs SNR', fontsize=13, fontweight='bold')
    axes2[-1].set_xlabel('SNR (dB)', fontsize=11, fontweight='bold')
    axes2[-1].set_ylabel('Correlation', fontsize=11, fontweight='bold')
    axes2[-1].grid(True, alpha=0.3, linestyle='--')
    axes2[-1].set_ylim(0, 1.05)
    
    # Beautify axes
    axes2[-1].spines['top'].set_visible(False)
    axes2[-1].spines['right'].set_visible(False)
    axes2[-1].spines['left'].set_linewidth(1.5)
    axes2[-1].spines['bottom'].set_linewidth(1.5)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save images
    signal_info_filename = "_".join(signal_names) if signal_names else f"{num_sources}sources"
    
    # Save combined plot
    fig.savefig(f'/root/IQUMamba1D/results/{results_folder}/saved_plots/correlation_vs_snr_combined_{signal_info_filename}.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save separate plots
    fig2.savefig(f'/root/IQUMamba1D/results/{results_folder}/saved_plots/correlation_vs_snr_individual_{signal_info_filename}.png', 
                 dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"Enhanced correlation plots saved:")
    print(f"  Combined: /root/IQUMamba1D/results/{results_folder}/saved_plots/correlation_vs_snr_combined_{signal_info_filename}.png")
    print(f"  Individual: /root/IQUMamba1D/results/{results_folder}/saved_plots/correlation_vs_snr_individual_{signal_info_filename}.png")