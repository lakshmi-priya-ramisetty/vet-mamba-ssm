# Vet-Mamba

This repository contains the `Vet-Mamba` project, which leverages the [Mamba State Space Model (SSM)](https://github.com/state-spaces/mamba) to create a specialized veterinary model. This project utilizes Mamba's efficient architecture, well-suited for handling information-dense data in a way that exceeds previous subquadratic models in language modeling.

## Table of Contents
- [About](#about)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About
Mamba is a state space model architecture optimized for tasks like language modeling, where it handles information-dense data more effectively than previous models. Built on structured state space model research, Mamba incorporates a hardware-aware design inspired by FlashAttention, providing efficient, scalable training. This project fine-tunes Mamba specifically for veterinary applications.

## Features
- **Training from Scratch on Large Datasets**: Capable of training from scratch on large datasets (up to 500 GB), enabling the model to learn domain-specific knowledge from vast veterinary data sources.
- **Veterinary-Specific Fine-Tuning**: Adapt the Mamba model for veterinary data and question answering.
- **Selective State Space Modeling**: Utilizes Mamba’s selective SSM layer to manage dense data effectively.
- **Mamba Block**: The core module wraps the selective SSM, allowing easy configuration and optimization for custom tasks.
- **Checkpointing and Logging**: Save training progress and monitor model performance.

## Demo
You can try our VetMamba model [here](https://8542-129-98-40-240.ngrok-free.app/).

## Requirements
- **Operating System**: Linux
- **GPU**: NVIDIA GPU (recommended); for AMD support, additional prerequisites are needed
- **Python**: 3.8 or later
- **PyTorch**: 1.12+
- **CUDA**: 11.6+
- **Dependencies**:
  - `torch==2.1.0`
  - `torchvision==0.16.0`
  - `torchaudio==2.1.0`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/lakshmi-priya-ramisetty/vet-mamba-ssm.git
   cd mamba
2. Install the core dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Choose one of the following options to install the Mamba package:

   - **Core Mamba Package**:
     ```bash
     pip install mamba-ssm
     ```

   - **Causal Conv1D Layer**: An efficient implementation of a simple causal Conv1D layer used inside the Mamba block. To install it separately:
     ```bash
     pip install causal-conv1d>=1.4.0
     ```

   - **Core Mamba Package with Causal Conv1D**: Install the core Mamba package along with the causal Conv1D layer.
     ```bash
     pip install mamba-ssm[causal-conv1d]
     ```

   - **Development Installation**: Install the core Mamba package with additional dependencies for development.
     ```bash
     pip install mamba-ssm[dev]
     ```

   > **Note**: If you encounter issues with PyTorch version compatibility, try using the `--no-build-isolation` flag with `pip`.

Refer to the [Mamba repository](https://github.com/state-spaces/mamba) for further details on installation and setup.

## Usage

To work with the `Vet-Mamba-SSM` project, you will primarily use two scripts:

1. **Model Training**: Use `train_mamba.py` to train the Mamba model from scratch on your veterinary dataset.
2. **Fine-Tuning for Question-Answering**: Use `finetune_mamba.py` to fine-tune the trained model on question-answering (QA) pairs.

### 1. Model Training

The `train.py` script is used to initialize and train the Mamba model on a veterinary-specific dataset. To start training, navigate to the `mamba_ssm/models` directory and run:

```bash
python mamba_ssm/models/train.py
```

### 2. Fine-Tuning for Question-Answering (QA)

After training the model, use `finetune_mamba.py` to fine-tune it on QA pairs to improve its performance in question-answering tasks. This step ensures the model can handle specific veterinary queries accurately.

```bash
python mamba_ssm/models/finetune_mamba.py 
```

### 3. VetMamba Configuration

The configuration used in this project is designed to handle large vocabulary sizes and dense layers suitable for long-sequence modeling.

- **Backbone (MixerModel)**: The backbone contains an embedding layer for token representations and a stack of 48 `Mamba Block` layers.
  - **Mixer (Mamba)**: Each `Block` includes a `Mamba` mixer with multiple linear projections and a `Conv1D` layer, designed for efficient processing of sequential data.
  - **Normalization (MambaRMSNorm)**: Each block and the final layer include normalization for stable training.

- **Language Model Head (`lm_head`)**: A linear layer that maps the 2048-dimensional hidden state to the 50,280-dimensional vocabulary space, enabling token prediction.

This configuration allows the Mamba model to efficiently handle long veterinary-related sequences, making it suitable for specialized tasks in the veterinary domain.

### 4. DeepSpeed Configuration for Optimized Training

The project uses DeepSpeed for optimized training, including mixed precision and large-scale model handling. Here’s an overview of the DeepSpeed configuration:

```json
deepspeed = {
      "comms_logger": {
          "enabled": True,
          "debug": True
      },
      "fp16": {
          "enabled": "auto",
          "loss_scale": 0,
          "loss_scale_window": 1000,
          "initial_scale_power": 16,
          "hysteresis": 2,
          "min_loss_scale": 1
      },

      "bf16": {
          "enabled": "auto"
      },

      "optimizer": {
          "type": "AdamW",
          "params": {
              "lr": 1e-4,
              "betas": [0.9, 0.999],
              "eps": 1e-8,
              "weight_decay": 0.01
          }
      },

      "scheduler": {
          "type": "WarmupDecayLR",
          "params": {
              "warmup_min_lr": 0.0,
              "warmup_max_lr": 1e-4,
              "warmup_num_steps": 10000,
              "total_num_steps": 500000,
          }
      },

      "zero_optimization": {
          "stage": 3,
          "offload_optimizer": {
              "device": "cpu",
              "pin_memory": True
          },
          "offload_param": {
              "device": "cpu",
              "pin_memory": True
          },
          "overlap_comm": True,
          "contiguous_gradients": True,
          "sub_group_size": 1e8,
          "reduce_bucket_size": "auto",
          "stage3_prefetch_bucket_size": "auto",
          "stage3_param_persistence_threshold": "auto",
          "stage3_max_live_parameters": 1e8,
          "stage3_max_reuse_distance": 1e8,
          "stage3_gather_16bit_weights_on_model_save": True
      },

      "gradient_accumulation_steps": 1,
      "gradient_clipping": 1.0,
      "steps_per_print": 2000,
      "train_batch_size": 1024, # Batch size
      "train_micro_batch_size_per_gpu": 256, # Batch size per GPU (4 GPUs)
      "wall_clock_breakdown": False
  }
```

- **Mixed Precision**: Both `fp16` and `bf16` are enabled for automatic mixed-precision training, which accelerates training while reducing memory usage.
- **Optimizer**: Uses `AdamW` with adaptive learning rate, betas, and epsilon values for efficient training.
- **Scheduler**: `WarmupDecayLR` scheduler gradually adjusts learning rate, providing a warmup phase for better convergence.
- **Zero Optimization (Stage 3)**: Enables CPU offloading for both parameters and optimizers, allowing efficient memory management for large models. It also supports overlapping communication and gradient contiguity to reduce memory fragmentation.
- **Gradient Accumulation and Clipping**: Allows control over gradient accumulation and clipping, which is essential for stability during training.

This DeepSpeed configuration is tailored to optimize the training process for large datasets and complex models like the Mamba SSM, ensuring efficient memory usage, faster convergence, and improved model performance.