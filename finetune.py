import os
import gc
import yaml
import argparse
import time
import signal
import sys
import torch
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import TrainerCallback, TrainerState, TrainerControl


@dataclass
class EarlyStopConfig:
    """Configuration for early stopping mechanisms"""
    enable_loss_early_stopping: bool = False
    patience: int = 5
    min_delta: float = 0.001
    min_steps: int = 100
    check_interval: int = 10
    target_loss: Optional[float] = None
    max_time_minutes: Optional[int] = None
    max_steps: Optional[int] = None


class SmartEarlyStoppingCallback(TrainerCallback):
    """Advanced early stopping callback with multiple stop conditions"""
    
    def __init__(self, config: EarlyStopConfig):
        self.config = config
        self.loss_history: List[float] = []
        self.best_loss = float('inf')
        self.patience_count = 0
        self.start_time = time.time()
        self.should_stop = False
        self.stop_reason = ""
        
        print("🧠 Smart Early Stopping initialized:")
        if config.target_loss:
            print(f"  🎯 Target loss: {config.target_loss}")
        if config.max_time_minutes:
            print(f"  ⏰ Max time: {config.max_time_minutes} minutes")
        if config.enable_loss_early_stopping:
            print(f"  📉 Loss patience: {config.patience} checks")
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Check stopping conditions on each log"""
        if logs is None or state.global_step < self.config.min_steps:
            return
        
        current_loss = logs.get('loss', float('inf'))
        
        # Check target loss
        if self._check_target_loss(current_loss, state.global_step):
            control.should_training_stop = True
            return
        
        # Check time limit
        if self._check_time_limit():
            control.should_training_stop = True
            return
        
        # Check max steps
        if self._check_max_steps(state.global_step):
            control.should_training_stop = True
            return
        
        # Check loss plateau (only at intervals)
        if state.global_step % self.config.check_interval == 0:
            if self._check_loss_plateau(current_loss, state.global_step):
                control.should_training_stop = True
                return
    
    def _check_target_loss(self, current_loss: float, step: int) -> bool:
        """Check if target loss is reached"""
        if self.config.target_loss and current_loss <= self.config.target_loss:
            self.stop_reason = f"🎯 Target loss {self.config.target_loss} reached! Current: {current_loss:.4f} at step {step}"
            print(f"\n{self.stop_reason}")
            return True
        return False
    
    def _check_time_limit(self) -> bool:
        """Check if time limit is exceeded"""
        if self.config.max_time_minutes:
            elapsed_minutes = (time.time() - self.start_time) / 60
            if elapsed_minutes >= self.config.max_time_minutes:
                self.stop_reason = f"⏰ Time limit reached: {elapsed_minutes:.1f}/{self.config.max_time_minutes} minutes"
                print(f"\n{self.stop_reason}")
                return True
        return False
    
    def _check_max_steps(self, step: int) -> bool:
        """Check if max steps reached"""
        if self.config.max_steps and step >= self.config.max_steps:
            self.stop_reason = f"🔢 Max steps reached: {step}/{self.config.max_steps}"
            print(f"\n{self.stop_reason}")
            return True
        return False
    
    def _check_loss_plateau(self, current_loss: float, step: int) -> bool:
        """Check if loss has plateaued"""
        if not self.config.enable_loss_early_stopping:
            return False
        
        self.loss_history.append(current_loss)
        
        # Keep history manageable
        if len(self.loss_history) > self.config.patience * 3:
            self.loss_history = self.loss_history[-self.config.patience * 2:]
        
        # Check for improvement
        if current_loss < (self.best_loss - self.config.min_delta):
            self.best_loss = current_loss
            self.patience_count = 0
            print(f"🔥 New best loss: {current_loss:.4f} at step {step}")
        else:
            self.patience_count += 1
            if self.patience_count % 2 == 0:  # Don't spam logs
                print(f"⏳ No improvement: {self.patience_count}/{self.config.patience} checks (best: {self.best_loss:.4f})")
        
        # Stop if patience exceeded
        if self.patience_count >= self.config.patience:
            self.stop_reason = f"📉 Early stopping: No improvement for {self.config.patience} checks (best loss: {self.best_loss:.4f})"
            print(f"\n{self.stop_reason}")
            return True
        
        return False


class GracefulKiller:
    """Handle Ctrl+C gracefully"""
    
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self._exit_gracefully)
        signal.signal(signal.SIGTERM, self._exit_gracefully)
    
    def _exit_gracefully(self, *args):
        print("\n🛑 Graceful shutdown initiated...")
        self.kill_now = True


class ConfigManager:
    """Enhanced configuration manager with validation and type conversion"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_and_validate()
    
    def _load_and_validate(self) -> Dict[str, Any]:
        """Load and validate configuration"""
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['model', 'training', 'dataset', 'output']
        missing = [s for s in required_sections if s not in config]
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")
        
        # Convert and validate types
        self._convert_types(config)
        return config
    
    def _convert_types(self, config: Dict[str, Any]):
        """Convert string values to appropriate types"""
        # Training config type conversions
        training = config.get('training', {})
        training_conversions = {
            'per_device_train_batch_size': int,
            'gradient_accumulation_steps': int,
            'num_train_epochs': int,
            'learning_rate': float,
            'warmup_ratio': float,
            'warmup_steps': int,
            'logging_steps': int,
            'save_steps': int,
            'save_total_limit': int,
            'max_steps': int,
        }
        
        for key, type_func in training_conversions.items():
            if key in training and isinstance(training[key], str):
                try:
                    training[key] = type_func(training[key])
                except (ValueError, TypeError):
                    print(f"⚠️ Warning: Could not convert {key} to {type_func.__name__}")
        
        # Model config type conversions
        model = config.get('model', {})
        if 'max_seq_length' in model and isinstance(model['max_seq_length'], str):
            try:
                model['max_seq_length'] = int(model['max_seq_length'])
            except (ValueError, TypeError):
                pass
        
        # LoRA config type conversions
        lora = model.get('lora', {})
        lora_conversions = {
            'r': int, 'alpha': int, 'dropout': float, 'random_state': int
        }
        for key, type_func in lora_conversions.items():
            if key in lora and isinstance(lora[key], str):
                try:
                    lora[key] = type_func(lora[key])
                except (ValueError, TypeError):
                    pass


class DatasetManager:
    """Handle dataset loading and preprocessing"""
    
    @staticmethod
    def format_conversations(examples, format_type: str = "chatml"):
        """Format conversations for training"""
        texts = []
        
        if format_type == "chatml":
            for conversation in examples.get("messages", []):
                text = ""
                for message in conversation:
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                texts.append(text)
        
        elif format_type == "alpaca":
            for conversation in examples.get("messages", []):
                if len(conversation) >= 2:
                    instruction = conversation[0].get("content", "")
                    response = conversation[1].get("content", "")
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}\n"
                    texts.append(text)
        
        elif format_type == "simple_qa":
            for item in examples.get("conversations", examples.get("messages", [])):
                if isinstance(item, dict):
                    question = item.get("question", item.get("input", ""))
                    answer = item.get("answer", item.get("output", ""))
                    text = f"Question: {question}\nAnswer: {answer}\n"
                    texts.append(text)
        
        return {"text": texts}
    
    @staticmethod
    def load_and_prepare_dataset(dataset_config: Dict[str, Any]):
        """Load and prepare dataset"""
        dataset_path = dataset_config['path']
        dataset_format = dataset_config.get('format', 'jsonl')
        
        print(f"📚 Loading dataset: {dataset_path}")
        
        # Load dataset
        if dataset_format in ['json', 'jsonl']:
            dataset = load_dataset("json", data_files=dataset_path)["train"]
        elif dataset_format == 'csv':
            dataset = load_dataset("csv", data_files=dataset_path)["train"]
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")
        
        print(f"📊 Original dataset size: {len(dataset)}")
        
        # Format conversations
        conversation_format = dataset_config.get('conversation_format', 'chatml')
        dataset = dataset.map(
            lambda x: DatasetManager.format_conversations(x, conversation_format),
            batched=True,
            desc="Formatting conversations"
        )
        
        # Apply filters
        if 'max_length' in dataset_config:
            max_length = dataset_config['max_length']
            original_size = len(dataset)
            dataset = dataset.filter(lambda x: len(x['text']) <= max_length)
            print(f"🔍 Filtered by length: {original_size} → {len(dataset)}")
        
        # Take subset if specified
        if 'subset_size' in dataset_config and dataset_config['subset_size'] > 0:
            subset_size = min(dataset_config['subset_size'], len(dataset))
            dataset = dataset.select(range(subset_size))
            print(f"📋 Using subset: {subset_size} samples")
        
        print(f"✅ Final dataset size: {len(dataset)}")
        
        # Show sample
        if len(dataset) > 0:
            sample_text = dataset[0]['text']
            print(f"📝 Sample text preview:\n{sample_text[:300]}{'...' if len(sample_text) > 300 else ''}")
        
        return dataset


class ModelManager:
    """Handle model loading and configuration"""
    
    @staticmethod
    def load_model_and_tokenizer(model_config: Dict[str, Any], hardware_config: Dict[str, Any]):
        """Load model with quantization and LoRA"""
        model_name = model_config['name']
        print(f"🤖 Loading model: {model_name}")
        
        # Load base model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=model_config.get('max_seq_length', 2048),
            dtype=model_config.get('dtype', None),
            load_in_4bit=model_config.get('load_in_4bit', True),
            trust_remote_code=model_config.get('trust_remote_code', True),
            use_cache=model_config.get('use_cache', False),
            device_map=hardware_config.get('device_map', "auto"),
        )
        
        # Add LoRA adapters
        print("🔗 Adding LoRA adapters...")
        lora_config = model_config.get('lora', {})
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config.get('r', 16),
            target_modules=lora_config.get('target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_alpha=lora_config.get('alpha', 16),
            lora_dropout=lora_config.get('dropout', 0.0),
            bias=lora_config.get('bias', "none"),
            use_gradient_checkpointing=lora_config.get('gradient_checkpointing', "unsloth"),
            random_state=lora_config.get('random_state', 42),
            use_rslora=lora_config.get('use_rslora', False),
            loftq_config=lora_config.get('loftq_config', None),
        )
        
        return model, tokenizer


class TrainingManager:
    """Handle training configuration and execution"""
    
    @staticmethod
    def create_training_config(training_config: Dict[str, Any], model_config: Dict[str, Any], 
                             output_dir: str) -> SFTConfig:
        """Create optimized training configuration"""
        
        return SFTConfig(
            # Core settings
            per_device_train_batch_size=training_config.get('per_device_train_batch_size', 2),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
            num_train_epochs=training_config.get('num_train_epochs', 1),
            max_steps=training_config.get('max_steps', -1),
            learning_rate=training_config.get('learning_rate', 2e-4),
            
            # Memory optimization
            gradient_checkpointing=training_config.get('gradient_checkpointing', True),
            optim=training_config.get('optimizer', "adamw_8bit"),
            
            # Precision
            bf16=training_config.get('bf16', True),
            fp16=training_config.get('fp16', False),
            
            # Scheduling
            warmup_steps=training_config.get('warmup_steps', 10),
            warmup_ratio=training_config.get('warmup_ratio', 0.1),
            lr_scheduler_type=training_config.get('lr_scheduler_type', "cosine"),
            
            # Logging and saving
            logging_steps=training_config.get('logging_steps', 25),
            logging_first_step=True,
            output_dir=output_dir,
            report_to=training_config.get('report_to', "none"),
            
            save_strategy=training_config.get('save_strategy', "steps"),
            save_steps=training_config.get('save_steps', 250),
            save_total_limit=training_config.get('save_total_limit', 3),
            
            # Dataset settings
            dataset_text_field="text",
            max_seq_length=model_config.get('max_seq_length', 2048),
            packing=training_config.get('packing', False),
            remove_unused_columns=True,
            dataloader_pin_memory=False,
            seed=training_config.get('seed', 42),
        )
    
    @staticmethod
    def create_trainer_with_callbacks(model, tokenizer, dataset, training_args, 
                                    early_stop_config: EarlyStopConfig):
        """Create trainer with smart callbacks"""
        callbacks = []
        
        # Add early stopping callback
        if any([
            early_stop_config.enable_loss_early_stopping,
            early_stop_config.target_loss,
            early_stop_config.max_time_minutes,
            early_stop_config.max_steps
        ]):
            callbacks.append(SmartEarlyStoppingCallback(early_stop_config))
        
        return SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            callbacks=callbacks,
        )


def setup_environment(hardware_config: Dict[str, Any]):
    """Setup training environment"""
    env_vars = hardware_config.get('environment_variables', {})
    default_env = {
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:256',
        'TOKENIZERS_PARALLELISM': 'false'
    }
    
    for key, value in {**default_env, **env_vars}.items():
        os.environ[key] = str(value)


def print_system_info():
    """Print system information"""
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = gpu_props.total_memory / 1024**3
        print(f"🔧 GPU: {gpu_props.name}")
        print(f"💾 VRAM: {allocated:.1f}GB / {total:.1f}GB")
        print(f"🆓 Free: {total - allocated:.1f}GB")


def test_model(model, tokenizer, test_prompts: List[str]):
    """Test the trained model"""
    if not test_prompts:
        return
    
    print("\n🧪 Testing the trained model...")
    FastLanguageModel.for_inference(model)
    
    for i, prompt in enumerate(test_prompts[:2], 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}")
        
        try:
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs, max_new_tokens=200, temperature=0.7, 
                    top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the response part
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1].strip()
            print(f"Response: {response}")
            
        except Exception as e:
            print(f"Test failed: {e}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Advanced Fine-tuning with Smart Early Stopping")
    parser.add_argument("--config", "-c", type=str, required=True, help="YAML configuration file")
    parser.add_argument("--output-dir", "-o", type=str, help="Override output directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    args = parser.parse_args()
    
    # Initialize graceful shutdown handler
    killer = GracefulKiller()
    
    try:
        # Load and validate configuration
        print("📋 Loading configuration...")
        config_manager = ConfigManager(args.config)
        config = config_manager.config
        
        # Extract configurations
        model_config = config['model']
        training_config = config['training']
        dataset_config = config['dataset']
        hardware_config = config.get('hardware', {})
        output_config = config['output']
        smart_training_config = config.get('smart_training', {})
        
        # Setup environment
        setup_environment(hardware_config)
        
        # Set output directory
        output_dir = args.output_dir or output_config.get('directory', './trained-model')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Starting advanced fine-tuning")
        print(f"📊 Config: {args.config}")
        print(f"📁 Output: {output_dir}")
        print(f"🤖 Model: {model_config['name']}")
        print(f"📚 Dataset: {dataset_config['path']}")
        
        if args.dry_run:
            print("✅ Configuration validation passed!")
            return
        
        # Load model and tokenizer
        model, tokenizer = ModelManager.load_model_and_tokenizer(model_config, hardware_config)
        
        # Prepare dataset
        dataset = DatasetManager.load_and_prepare_dataset(dataset_config)
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        print_system_info()
        
        # Create training configuration
        training_args = TrainingManager.create_training_config(
            training_config, model_config, output_dir
        )
        
        # Setup early stopping
        early_stop_config = EarlyStopConfig(
            enable_loss_early_stopping=smart_training_config.get('enable_loss_early_stopping', False),
            patience=smart_training_config.get('early_stop_patience', 5),
            min_delta=smart_training_config.get('early_stop_min_delta', 0.001),
            min_steps=smart_training_config.get('early_stop_min_steps', 100),
            check_interval=smart_training_config.get('early_stop_check_interval', 10),
            target_loss=smart_training_config.get('target_loss'),
            max_time_minutes=smart_training_config.get('max_time_minutes'),
            max_steps=smart_training_config.get('max_steps')
        )
        
        # Create trainer
        trainer = TrainingManager.create_trainer_with_callbacks(
            model, tokenizer, dataset, training_args, early_stop_config
        )
        
        # Start training
        print("\n🏋️ Training started...")
        print("💡 Press Ctrl+C for graceful shutdown")
        
        start_time = time.time()
        trainer_stats = None
        
        try:
            # Training loop with graceful shutdown check
            trainer_stats = trainer.train()
            
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user")
            
        # Check for graceful killer
        if killer.kill_now:
            print("\n🛑 Graceful shutdown requested")
        
        # Always try to save the model
        print("\n💾 Saving model...")
        try:
            save_method = output_config.get('save_method', 'merged_16bit')
            if save_method in ['merged_16bit', 'merged_4bit', 'lora']:
                model.save_pretrained_merged(output_dir, tokenizer, save_method=save_method)
            else:
                trainer.save_model()
            print(f"✅ Model saved to {output_dir}")
        except Exception as e:
            print(f"⚠️ Save error: {e}")
            trainer.save_model()  # Fallback
        
        # Print training summary
        elapsed_time = (time.time() - start_time) / 60
        print(f"\n📊 Training Summary:")
        print(f"⏱️ Training time: {elapsed_time:.1f} minutes")
        
        if trainer_stats and hasattr(trainer_stats, 'training_loss'):
            print(f"📉 Final loss: {trainer_stats.training_loss:.4f}")
        
        # Test the model
        test_prompts = output_config.get('test_prompts', [])
        if test_prompts:
            test_model(model, tokenizer, test_prompts)
        
        # Save configuration for reference
        config_save_path = Path(output_dir) / "training_config.yaml"
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"📋 Config saved to {config_save_path}")
        
        print("\n🎉 Fine-tuning completed successfully!")
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted")
        sys.exit(0)
        
    except torch.cuda.OutOfMemoryError:
        print("\n❌ GPU Out of Memory!")
        print("💡 Suggestions:")
        print("  - Reduce batch_size")
        print("  - Increase gradient_accumulation_steps")
        print("  - Reduce max_seq_length")
        print("  - Use smaller model")
        torch.cuda.empty_cache()
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()