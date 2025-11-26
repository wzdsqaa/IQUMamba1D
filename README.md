# IQUMamba-1D: A Mamba-Enhanced 1D U-Net for Single-Channel Communication Signal Blind Source Separation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official code repository for the paper:

**[Paper Title: IQUMamba-1D: A Mamba-Enhanced 1D U-Net for Single-Channel Communication Signal Blind Source Separation]**

**Authors:** 


> **Abstract:** Single-channel blind source separation (SCBSS) remains a formidable challenge in modern wireless communication systems, especially when applied to long-duration, spectrally overlapped complex baseband signals. Existing deep learning approaches are hindered by limited receptive fields, vanishing gradients, or quadratic computational complexity, making them impractical for real-time separation of extended signal frames. To overcome these limitations, we propose IQUMamba-1D, a one-dimensional U-Net architecture enhanced with selective structured state-space model (SSM). The framework synergistically integrates long-range temporal modeling through Mamba blocks while preserving multi-scale feature extraction via U-Net. Three communication-specific innovations are introduced: adaptive dual tokenization mitigates quadrature imbalance by stage-dependent tokenization mode selection between temporal and channel token modes; complex-valued selective state space modeling enforces geometric consistency in the complex plane by jointly processing in-phase and quadrature components; and an adaptive skip connection processor alleviates semantic gaps induced by aggressive downsampling, thereby preserving phase coherence vital for signal reconstruction. Evaluated across synthetic and public datasets with signal lengths up to 32 768 samples, IQUMamba-1D achieves state-of-the-art separation performance with linear computational complexity. The proposed method provides a theoretically grounded and computationally efficient solution for SCBSS of spectrally overlapped communication signals under single-channel observation.


## Introduction

IQUMamba-1D is designed to address the challenges of separating spectrally overlapped communication signals from a single observation channel. It leverages the linear-time modeling capabilities of Mamba blocks within a U-Net architecture to handle long sequences effectively. Key features include:

*   **Long-Range Dependency Modeling:** Utilizes Mamba blocks to capture dependencies over long signal sequences (e.g., up to 32768 samples).
*   **Phase Coherence Preservation:** Ensures the integrity of phase information crucial for signal reconstruction.
*   **Linear Complexity:** Offers efficient computation suitable for processing long-duration signals.
*   **Effective Separation:** Achieves state-of-the-art performance on benchmarks for communication signal separation.

This repository contains the code necessary to train, evaluate, and reproduce the results reported in the paper.

## Requirements

*   **Operating System:** Ubuntu 22.04 (or similar Linux environment recommended)
*   **Python:** `>= 3.12` (tested with `3.12`)
*   **CUDA:** `>= 12.4` (for GPU acceleration)
*   **PyTorch:** `>= 2.5.1` (compatible with CUDA version)

*   Other dependencies are listed in `requirements.txt`.

## Data Preparation

1.  **Download Datasets:** Obtain the required datasets (e.g., RML2016.10a, RML2018.01a, TorchSig, 8PSK-E, or your custom dataset).
2.  **Organize Data:** Public dataset BPSK and QPSK waveform files: You can download the RML2016.10a or RML2018.01a dataset yourself, use the TorchSig toolkit to generate BPSK and QPSK data, and convert the data at the corresponding signal-to-noise ratio into mat files. You can also use our pre-processed data at `data/` directory. The dataset generation code used in this article is located in the `dataset_gen`.
3.  **Prepare Data Format:** Ensure your data loading scripts (e.g., in `main.py` or a dedicated data loader script) can read the data from the specified paths and format it correctly for the model (e.g., complex-valued I/Q signals).

## Usage

### Training and Evaluation

To train the IQUMamba-1D model, use the `main.py` script with appropriate arguments. Example commands can be found in the `run.bash` file provided.

*   **Basic Training Example:**
    ```bash
    python main.py --data_choice [DATASET_NAME] --source_names [SOURCE1] [SOURCE2] --other_args ...
    ```
    *   `--data_choice`: Specify the dataset (e.g., `2016`, `2018`, `TorchSig`, or a custom identifier).
    *   `--source_names`: Specify the signal types to be separated (e.g., `BPSK QPSK`).
    *   `--mode`: Specify the training mode (e.g., `train`, `test`).
    *   `--other_args`: Include other necessary arguments like loss function, stages, random seed, etc. Refer to the argument parser in `main.py`.

*   **Example from `run.bash`:**
    ```bash
    # Example command from run.bash (TorchSig dataset, multiple runs)
    python main.py --data_choice TorchSig --source_names BPSK QPSK --multiple_runs --num_runs 5 --start_seed 42
    ```

## Acknowledgements

*   We thank the authors of the [U-Mamba repository](https://github.com/bowang-lab/U-Mamba) for making their U-Net and Mamba integration code publicly available. The implementation of our 1D U-Net architecture in `models/IQUMamba.py` was inspired by and adapted from their work.

## Citation

If you find this code or paper useful, please cite:

```bibtex
@article{IQUMamba1D,
  title={IQUMamba-1D: A Mamba-Enhanced 1D U-Net for Single-Channel Communication Signal Blind Source Separation},
}