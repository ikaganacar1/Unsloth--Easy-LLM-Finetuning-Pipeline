# Unsloth Fine-tuning Pipeline

A production-ready pipeline for fine-tuning large language models using Unsloth with YAML-based configuration, advanced training features, and web-based model serving.

## Overview

This pipeline provides a streamlined approach to fine-tuning language models with:

- Single YAML configuration file for all training parameters
- Smart early stopping with multiple stop conditions
- Memory-optimized training with 4-bit quantization
- LoRA (Low-Rank Adaptation) fine-tuning support
- ChatML conversation format support
- Graceful interruption handling
- Built-in model testing capabilities
- **Web-based model serving with Gradio interface**

## Tested Configuration

- **Model**: unsloth/Qwen3-8B-unsloth-bnb-4bit
- **Hardware**: 12GB RTX 5070 GPU
- **Dataset Format**: ChatML (tested and validated)
- **Training**: Successfully completed with provided configuration
- **Serving**: Gradio web interface for interactive model testing

## Installation

### Requirements

Install dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

### Manual Installation

```bash
pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
pip install torch torchvision torchaudio
pip install transformers datasets trl peft accelerate bitsandbytes
pip install PyYAML gradio
```

## Quick Start

### Training

1. **Prepare your dataset** in ChatML format (JSONL file)
2. **Configure training parameters** in `config.yaml`
3. **Start training**:

```bash
python finetune.py --config config.yaml
```

### Model Serving

After training, serve your model with a web interface:

```bash
python launch_model.py
```

This will start a Gradio interface at `http://localhost:7860` where you can interact with your fine-tuned model.

### Dataset Format

The pipeline expects JSONL files with ChatML conversation format:

```json
{
  "messages": [
    {"role": "user", "content": "Your question here"},
    {"role": "assistant", "content": "Model response here"}
  ]
}
```

## Configuration Guide

The `config.yaml` file is divided into several sections, each controlling different aspects of the training process.

### Model Configuration

```yaml
model:
  name: "unsloth/Qwen3-8B-unsloth-bnb-4bit"
  max_seq_length: 4096
  dtype: null
  load_in_4bit: true
  trust_remote_code: true
  use_cache: false
```

**Parameters:**
- `name`: Hugging Face model identifier
- `max_seq_length`: Maximum sequence length for training
- `dtype`: Data type (null for auto-detection)
- `load_in_4bit`: Enable 4-bit quantization for memory efficiency
- `trust_remote_code`: Allow custom model code execution
- `use_cache`: Enable/disable model caching

### LoRA Configuration

```yaml
  lora:
    r: 8
    alpha: 8
    dropout: 0.0
    bias: "none"
    target_modules: [
      "q_proj", "k_proj", "v_proj", "o_proj",
      "gate_proj", "up_proj", "down_proj"
    ]
    gradient_checkpointing: "unsloth"
    random_state: 42
    use_rslora: true
    loftq_config: null
```

**Parameters:**
- `r`: LoRA rank (dimensionality of adaptation)
- `alpha`: LoRA alpha parameter for scaling
- `dropout`: Dropout rate for LoRA layers
- `bias`: Bias handling ("none", "all", or "lora_only")
- `target_modules`: List of modules to apply LoRA adaptation
- `gradient_checkpointing`: Checkpointing strategy
- `random_state`: Random seed for reproducibility
- `use_rslora`: Enable rank-stabilized LoRA
- `loftq_config`: LoftQ configuration (null to disable)

### Training Configuration

```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  num_train_epochs: 1
  learning_rate: 0.0001
  max_steps: -1
  
  gradient_checkpointing: true
  optimizer: "adamw_8bit"
  bf16: true
  fp16: false
  
  warmup_steps: 1000
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  
  logging_steps: 100
  save_strategy: "steps"
  save_steps: 1000
  save_total_limit: 10
  report_to: "none"
  
  packing: false
  seed: 42
```

**Core Training Parameters:**
- `per_device_train_batch_size`: Batch size per GPU device
- `gradient_accumulation_steps`: Steps to accumulate gradients
- `num_train_epochs`: Number of training epochs
- `learning_rate`: Initial learning rate
- `max_steps`: Maximum training steps (-1 for epoch-based)

**Optimization Settings:**
- `gradient_checkpointing`: Enable gradient checkpointing for memory efficiency
- `optimizer`: Optimizer type ("adamw_8bit" for memory efficiency)
- `bf16`/`fp16`: Mixed precision training settings

**Scheduling:**
- `warmup_steps`: Number of warmup steps
- `warmup_ratio`: Warmup ratio (alternative to warmup_steps)
- `lr_scheduler_type`: Learning rate scheduler type

**Logging and Saving:**
- `logging_steps`: Frequency of logging
- `save_strategy`: When to save checkpoints ("steps" or "epoch")
- `save_steps`: Save frequency in steps
- `save_total_limit`: Maximum number of checkpoints to keep

### Dataset Configuration

```yaml
dataset:
  path: "comprehensive_dataset.jsonl"
  format: "jsonl"
  conversation_format: "chatml"
  max_length: 4096
  subset_size: 0
```

**Parameters:**
- `path`: Path to the training dataset
- `format`: Dataset file format ("jsonl", "json", "csv")
- `conversation_format`: Format for conversation data ("chatml", "alpaca")
- `max_length`: Maximum sequence length for filtering
- `subset_size`: Use subset of data (0 for full dataset)

### Hardware Configuration

```yaml
hardware:
  device_map: "auto"
  environment_variables:
    PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:256"
    TOKENIZERS_PARALLELISM: "false"
    CUDA_VISIBLE_DEVICES: "0"
```

**Parameters:**
- `device_map`: Device mapping strategy ("auto" for automatic)
- `environment_variables`: CUDA and training environment variables

### Smart Training Configuration

```yaml
smart_training:
  enable_loss_early_stopping: true
  early_stop_patience: 30
  early_stop_min_delta: 0.005
  early_stop_min_steps: 300
  early_stop_check_interval: 25
  
  target_loss: 0.3
  max_time_minutes: 3000000
  max_steps: null
  
  dataset_num_proc: 4
```

**Early Stopping Parameters:**
- `enable_loss_early_stopping`: Enable loss plateau detection
- `early_stop_patience`: Steps to wait without improvement
- `early_stop_min_delta`: Minimum improvement threshold
- `early_stop_min_steps`: Minimum steps before early stopping
- `early_stop_check_interval`: Frequency of early stopping checks

**Training Limits:**
- `target_loss`: Stop training when reaching target loss
- `max_time_minutes`: Maximum training time in minutes
- `max_steps`: Maximum training steps override

### Output Configuration

```yaml
output:
  directory: "./qwen-kubernetes-0.0.8"
  save_method: "merged_16bit"
  
  test_prompts:
    - "How do I create a Kubernetes deployment with 3 replicas and resource limits?"
    - "What's the difference between a Service and an Ingress in Kubernetes?"
    - "How can I debug a pod that's stuck in Pending state?"
    - "Explain Kubernetes ConfigMaps and Secrets with examples"
    - "How do I set up horizontal pod autoscaling based on CPU usage?"
```

**Parameters:**
- `directory`: Output directory for trained model
- `save_method`: Model saving format ("merged_16bit", "merged_4bit", "lora")
- `test_prompts`: List of prompts for post-training validation

## Model Serving with Gradio

### Launching the Web Interface

After training your model, you can serve it with an interactive web interface:

```bash
python launch_model.py
```

### Features

- **Interactive Chat Interface**: Real-time conversation with your fine-tuned model
- **Kubernetes Expertise**: Pre-configured for Kubernetes-related queries
- **Example Prompts**: Built-in examples to get started quickly
- **Responsive Design**: Works on desktop and mobile devices
- **Shareable Interface**: Option to create public links for sharing

### Customization

To use the serving script with your own model, update the model path in `launch_model.py`:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="path/to/your/trained/model",  # Update this path
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)
```

### Configuration Options

The Gradio interface supports several configuration options:

- **Temperature**: Controls response randomness (0.1-1.0)
- **Max Tokens**: Maximum response length
- **Repetition Penalty**: Prevents repetitive responses
- **System Prompt**: Customizable system instructions

## Advanced Features

### Smart Early Stopping

The pipeline includes multiple early stopping mechanisms:

- **Loss Plateau Detection**: Monitors training loss and stops when improvement plateaus
- **Target Loss Achievement**: Stops when a specific loss threshold is reached
- **Time-based Stopping**: Limits training to a maximum time duration
- **Step-based Stopping**: Limits training to a maximum number of steps

### Memory Optimization

- **4-bit Quantization**: Reduces memory usage significantly
- **Gradient Checkpointing**: Trades computation for memory
- **8-bit Optimizers**: Memory-efficient optimization algorithms
- **Automatic Memory Management**: Built-in garbage collection and cache clearing

### Graceful Interruption

Training can be safely interrupted (Ctrl+C) while preserving the current model state:

```bash
^C
Graceful shutdown initiated...
Saving model...
Model saved to ./output-directory
```

## Command Line Options

### Training

```bash
python finetune.py --config config.yaml [OPTIONS]
```

**Available Options:**
- `--config, -c`: Path to YAML configuration file (required)
- `--output-dir, -o`: Override output directory from config
- `--dry-run`: Validate configuration without starting training

### Model Serving

```bash
python launch_model.py
```

The script will automatically start the Gradio interface on `http://localhost:7860`.

## Hardware Requirements

### Minimum Requirements
- **GPU**: 12GB VRAM (tested on RTX 5070)
- **RAM**: 16GB system RAM
- **Storage**: 20GB free space

### Recommended Setup
- **GPU**: 16GB+ VRAM
- **RAM**: 32GB+ system RAM
- **Storage**: SSD with 50GB+ free space

### Serving Requirements
- **GPU**: 8GB+ VRAM (for inference)
- **RAM**: 8GB+ system RAM
- **Network**: Stable internet connection for web interface

## Troubleshooting

### Out of Memory Errors

Reduce memory usage by adjusting these parameters:

```yaml
training:
  per_device_train_batch_size: 1    # Reduce batch size
  gradient_accumulation_steps: 16   # Increase accumulation steps

model:
  max_seq_length: 2048             # Reduce sequence length
```

### Dataset Loading Issues

Ensure your JSONL file follows the correct format:

```bash
# Validate dataset structure
head -n 3 your_dataset.jsonl | python -m json.tool
```

### Model Serving Issues

If the Gradio interface fails to start:

1. **Check model path**: Ensure the model path in `launch_model.py` is correct
2. **GPU memory**: Make sure you have enough VRAM for inference
3. **Port conflicts**: Try a different port if 7860 is occupied
4. **Dependencies**: Ensure Gradio is installed: `pip install gradio>=4.0.0`

## Project Structure

```
unsloth-pipeline/
├── config.yaml                 # Main configuration file
├── finetune.py                 # Training script
├── launch_model.py             # Model serving script (NEW!)
├── requirements.txt            # Python dependencies (updated)
├── devops-42k/                # Dataset processing example
│   └── devops-42k.ipynb       # Data cleaning notebook
└── README.md                  # Documentation (updated)
```

## Usage Examples

### Complete Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your dataset (see devops-42k/ for example)
python process_dataset.py

# 3. Train the model
python finetune.py --config config.yaml

# 4. Serve the trained model
python launch_model.py
```

### Quick Test

```bash
# Test configuration without training
python finetune.py --config config.yaml --dry-run

# Train with custom output directory
python finetune.py --config config.yaml --output-dir ./my-model

# Serve with specific model
python launch_model.py  # Edit model path in script first
```