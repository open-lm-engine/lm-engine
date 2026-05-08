<!-- **************************************************
Copyright (c) 2025, Mayank Mishra
************************************************** -->

<h1 align="center">LM Engine</h1>

<p align="center">
  <b>A Hyper-Optimized Library for Pretraining, Finetuning, and Distillation of Large Language Models</b>
</p>

<p align="center">
  <a href="https://github.com/open-lm-engine/lm-engine/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.8+-orange.svg" alt="PyTorch 2.8+"></a>
  <a href="https://discord.gg/AFDxmjH5RV"><img src="https://img.shields.io/badge/Discord%20-blue.svg"></a>
</p>

---

## Overview

**LM Engine** is a research-grade, production-ready library for training large language models at scale. Built with performance and flexibility in mind, it provides native support for multiple accelerators including NVIDIA GPUs, Google TPUs, and AWS Trainiums.

### Key Features

- 🚀 **Multi-Accelerator Support** — Train on NVIDIA CUDA GPUs, Google Cloud TPUs, and AWS Trainium
- ⚡ **Advanced Distributed Training** — FSDP (1 & 2), Tensor Parallelism, Pipeline Parallelism, and ZeRO stages 1-3
- 🔧 **Flexible Model Architectures** — Transformer variants, MoE, Mamba2, RNNs, and hybrid architectures
- 📦 **HuggingFace Integration** — Seamless import/export with the HuggingFace ecosystem
- 🎯 **Training Modes** — Pretraining from scratch, full finetuning, and knowledge distillation
- 🔥 **Custom Kernels** — High-performance Triton, CUDA, and Pallas kernels via [XMA](./accelerated-model-architectures/)
- 📊 **Experiment Tracking** — Native Weights & Biases and Aim integration
- 💾 **Efficient Checkpointing** — Async checkpointing with full state resumability

---

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/open-lm-engine/lm-engine.git
cd lm-engine

# Install with uv
uv sync --extra cuda  # For NVIDIA GPUs
uv sync --extra tpu   # For Google TPUs
```

### Using pip

```bash
pip install lm-engine

# With optional dependencies
pip install "lm-engine[cuda]"      # NVIDIA GPU support
pip install "lm-engine[tpu]"       # Google TPU support
pip install "lm-engine[mamba2]"    # Mamba2 architecture support
pip install "lm-engine[data]"      # Data preprocessing utilities
pip install "lm-engine[dev]"       # Development dependencies
```

### Docker

```bash
# Build for TPU
docker build --build-arg EXTRA=tpu -t lm-engine:tpu -f docker/Dockerfile .

# Build for CUDA
docker build --build-arg EXTRA=cuda -t lm-engine:cuda -f docker/Dockerfile .
```

---

## Quick Start

### Pretraining

Launch training using a sample pretraining config in [configs folder](configs/).

```bash
# Single GPU
python -m lm_engine.pretrain --config config.yml

# Multi-GPU with torchrun
torchrun --nproc_per_node=8 -m lm_engine.pretrain --config config.yml

# Multi-node
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m lm_engine.pretrain --config config.yml
```

---

## Distributed Training

LM Engine provides comprehensive distributed training support:

### Data Parallelism (ZeRO)

```yaml
distributed_args:
  stage: 3  # ZeRO stage (0, 1, 2, or 3)
  fsdp_algorithm: 2  # FSDP-1 or FSDP-2
  zero_topology:
    data_parallel_replication_world_size: 2
    data_parallel_sharding_world_size: 4
```

### Tensor Parallelism

```yaml
distributed_args:
  tensor_parallel_world_size: 4
  sequence_parallel: true
  use_async_tensor_parallel: true
```

### Pipeline Parallelism

```yaml
distributed_args:
  pipeline_parallel_world_size: 2
  num_pipeline_stages: 4
  pipeline_parallel_schedule: "1F1B"
```

### Gradient Checkpointing

```yaml
distributed_args:
  gradient_checkpointing_method: block  # or "full"
  gradient_checkpointing_args:
    checkpoint_every_n_layers: 2
```

---

## Model Import/Export

### Import from HuggingFace

```python
from tools.import_from_hf import convert_hf_to_lm_engine

convert_hf_to_lm_engine(
    model_name="meta-llama/Llama-3.1-8B",
    output_path="./converted-model"
)
```

### Export to HuggingFace

```python
from tools.export_to_hf import convert_lm_engine_to_hf

convert_lm_engine_to_hf(
    checkpoint_path="./checkpoints/step-10000",
    output_path="./hf-model",
    model_type="llama"
)
```

### Unshard Checkpoints

```bash
python -m lm_engine.unshard --config unshard.yml
```

---

## Configuration Reference

<details>
<summary><b>Full Configuration Options</b></summary>

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_training_steps` | int | required | Total training steps |
| `micro_batch_size` | int | required | Batch size per device |
| `gradient_accumulation_steps` | int | 1 | Gradient accumulation steps |
| `gradient_clipping` | float | 1.0 | Max gradient norm |
| `eval_during_training` | bool | true | Enable validation |
| `eval_interval` | int | — | Steps between evaluations |

### Optimizer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_name` | str | "TorchAdamW" | Optimizer class |
| `lr` | float | 1e-5 | Learning rate |
| `weight_decay` | float | 0.1 | Weight decay |
| `betas` | list | [0.9, 0.95] | Adam betas |

### LR Scheduler

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr_decay_style` | str | "cosine" | Decay schedule (linear, cosine, exponential) |
| `num_warmup_steps` | int | 200 | Warmup steps |
| `num_decay_steps` | int | — | Decay steps (defaults to remaining) |
| `lr_decay_factor` | float | 0.1 | Final LR ratio |

### Mixed Precision

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dtype` | str | "fp32" | Training dtype (fp32, bf16, fp16) |

</details>

---

## Cloud Deployment

### Google Cloud TPU

```bash
# See scripts/gcp-tpu/ for TPU-specific launch scripts
./scripts/gcp-tpu/launch.sh --config config.yml --tpu-name my-tpu-v4
```

### AWS Trainium

```bash
# See scripts/aws-trainium/ for Trainium launch scripts
./scripts/aws-trainium/launch.sh --config config.yml
```

### Kubernetes (GKE)

```bash
# See scripts/gke/ for Kubernetes manifests
kubectl apply -f scripts/gke/training-job.yml
```

---

## Community

Join the [Discord server](https://discord.gg/AFDxmjH5RV) to discuss LLM architecture research, distributed training, and contribute to the project!

---

## Citation

If you use LM Engine in your research, please cite:

```bibtex
@software{mishra2024lmengine,
  title = {LM Engine: A Hyper-Optimized Library for Pretraining and Finetuning},
  author = {Mishra, Mayank},
  year = {2024},
  url = {https://github.com/open-lm-engine/lm-engine}
}
```

---

## License

LM Engine is released under the [Apache 2.0 License](LICENSE).

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/mayank31398">Mayank Mishra</a></sub>
</p>
