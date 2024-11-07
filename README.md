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
2. **Fine-Tuning for Question-Answering**: Use `mamba_finetune_results.py` to fine-tune the trained model on question-answering (QA) pairs.

### 1. Model Training

The `train_mamba.py` script is used to initialize and train the Mamba model on a veterinary-specific dataset. To start training, navigate to the `mamba_ssm/models` directory and run:

```bash
python mamba_ssm/models/train_mamba.py
```

### 2. Fine-Tuning for Question-Answering (QA)

After training the model, use `mamba_finetune_results.py` to fine-tune it on QA pairs to improve its performance in question-answering tasks. This step ensures the model can handle specific veterinary queries accurately.

```bash
python mamba_finetune_results.py 
```

### 3. MambaLMHeadModel Configuration

The `MambaLMHeadModel` configuration used in this project is designed to handle large vocabulary sizes and dense layers suitable for long-sequence modeling. Here’s an overview of its architecture:

```python
MambaLMHeadModel(
  (backbone): MixerModel(
    (embedding): Embedding(32000, 2048)
    (layers): ModuleList(
      (0-47): 48 x Block(
        (mixer): Mamba(
          (in_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (conv1d): Conv1d(4096, 4096, kernel_size=(4,), stride=(1,), padding=(3,), groups=4096)
          (act): SiLU()
          (x_proj): Linear(in_features=4096, out_features=160, bias=False)
          (dt_proj): Linear(in_features=128, out_features=4096, bias=True)
          (out_proj): Linear(in_features=4096, out_features=2048, bias=False)
        )
        (norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
      )
    )
    (norm_f): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=2048, out_features=32000, bias=False)
)
```

- **Backbone (MixerModel)**: The backbone contains an embedding layer for token representations and a stack of 48 `Block` layers.
  - **Mixer (Mamba)**: Each `Block` includes a `Mamba` mixer with multiple linear projections and a `Conv1D` layer, designed for efficient processing of sequential data.
  - **Normalization (LayerNorm)**: Each block and the final layer include layer normalization for stable training.

- **Language Model Head (`lm_head`)**: A linear layer that maps the 2048-dimensional hidden state to the 32,000-dimensional vocabulary space, enabling token prediction.

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
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto",
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
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}
```

- **Mixed Precision**: Both `fp16` and `bf16` are enabled for automatic mixed-precision training, which accelerates training while reducing memory usage.
- **Optimizer**: Uses `AdamW` with adaptive learning rate, betas, and epsilon values for efficient training.
- **Scheduler**: `WarmupDecayLR` scheduler gradually adjusts learning rate, providing a warmup phase for better convergence.
- **Zero Optimization (Stage 3)**: Enables CPU offloading for both parameters and optimizers, allowing efficient memory management for large models. It also supports overlapping communication and gradient contiguity to reduce memory fragmentation.
- **Gradient Accumulation and Clipping**: Allows control over gradient accumulation and clipping, which is essential for stability during training.

This DeepSpeed configuration is tailored to optimize the training process for large datasets and complex models like the Mamba SSM, ensuring efficient memory usage, faster convergence, and improved model performance.

### 5. Working on a 1.4 Billion Parameter Model

Currently, I am working on a larger model with 1.4 billion parameters using the `MambaForCausalLM` architecture. This configuration leverages Mamba’s state-space efficiency to handle extensive parameter sets effectively:

```python
MambaForCausalLM(
  (backbone): MambaModel(
    (embeddings): Embedding(50280, 2048)
    (layers): ModuleList(
      (0-47): 48 x MambaBlock(
        (norm): MambaRMSNorm(2048, eps=1e-05)
        (mixer): MambaMixer(
          (conv1d): Conv1d(4096, 4096, kernel_size=(4,), stride=(1,), padding=(3,), groups=4096)
          (act): SiLU()
          (in_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (x_proj): Linear(in_features=4096, out_features=160, bias=False)
          (dt_proj): Linear(in_features=128, out_features=4096, bias=True)
          (out_proj): Linear(in_features=4096, out_features=2048, bias=False)
        )
      )
    )
    (norm_f): MambaRMSNorm(2048, eps=1e-05)
  )
  (lm_head): Linear(in_features=2048, out_features=50280, bias=False)
)
```

This model configuration expands on the standard Mamba setup, using 48 `MambaBlock` layers with sophisticated normalization, projection, and activation functions. The `MambaForCausalLM` model is designed for large-scale causal language modeling, making it ideal for complex applications requiring high parameter counts. This architecture enables the handling of extensive data while preserving efficient processing speed and memory usage.
