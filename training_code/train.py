"""
Main training script for PosterMaker Stage 1 IDS-based pipeline.
This script orchestrates the complete training process including data loading,
model initialization, and training execution.
"""

import torch
import argparse
import os
import random
import numpy as np
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import yaml
import logging

logger = logging.getLogger(__name__)

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train PosterMaker Stage 1 with dual ControlNets")
    parser.add_argument(
        "--config_file", type=str, required=True, help="Path to the training configuration file.",
        default="configs/train_config.yaml"
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from."
    )
    return parser.parse_args()

# Import all our custom modules
from src.config import load_config
from src.dataset import PosterDatasetStage1, create_stage1_dataloader
from src.utils.ids_query import IDSQuery
from src.utils.ids_tokenizer import IDSTokenizer
from src.model_loader import load_all_models_for_stage1
from src.trainer import Trainer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


def create_scheduler(optimizer, config):
    """Create learning rate scheduler based on config."""
    stage1_config = config['stage1']
    scheduler_type = stage1_config.get('lr_scheduler_type', 'cosine')
    
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=stage1_config['num_train_epochs'],
            eta_min=stage1_config['learning_rate'] * 0.01
        )
    elif scheduler_type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=stage1_config.get('lr_step_size', 10),
            gamma=stage1_config.get('lr_gamma', 0.1)
        )
    else:
        print(f"Warning: Unknown scheduler type '{scheduler_type}', using cosine")
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=stage1_config['num_train_epochs'],
            eta_min=stage1_config['learning_rate'] * 0.01
        )
    
    return scheduler


def main():
    """Main training function"""
    args = parse_args()
    
    # --- 1. Load Configuration ---
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # --- 2. Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    # --- 3. Load Models ---
    print("Loading models...")
    models, tokenizers = load_all_models_for_stage1(config, device)
    
    # --- 4. Initialize IDS Tokenizer for Dataset ---
    # This tokenizer is used for data preprocessing, not for prompt encoding.
    
    # Get paths based on updated configuration
    if config['ids_database_path'].startswith('./'):
        # IDS database path is relative to training_code
        ids_database_path = config['ids_database_path']
    else:
        # IDS database path is relative to poster_maker_dir (legacy support)
        poster_maker_dir = config['poster_maker_dir']
        ids_database_path = os.path.join(poster_maker_dir, config['ids_database_path'])
    
    # Check for vocabulary file
    vocab_file = None
    if 'ids_vocab_path' in config:
        if config['ids_vocab_path'].startswith('./'):
            # Vocabulary path is relative to training_code
            vocab_path = config['ids_vocab_path']
        else:
            # Vocabulary path is relative to poster_maker_dir (legacy support)
            poster_maker_dir = config['poster_maker_dir']
            vocab_path = os.path.join(poster_maker_dir, config['ids_vocab_path'])
            
        if os.path.exists(vocab_path):
            vocab_file = vocab_path
            print(f"Using pre-built vocabulary: {vocab_file}")
        else:
            print(f"Vocabulary file not found at {vocab_path}, will build vocab from IDS database")
    
    # Initialize tokenizer with updated interface
    tokenizer = IDSTokenizer(
        ids_database_path=ids_database_path,
        vocab_file=vocab_file,
        preserve_rare_chars=True
    )
    print(f"Tokenizer initialized with vocabulary size: {tokenizer.vocab_size}")
    
    # --- 5. Load Data ---
    print("Initializing tokenizer and datasets...")
    
    # Create datasets using the helper function
    try:
        train_dataloader = create_stage1_dataloader(
            config, tokenizer, split='train',
            batch_size=config['stage1']['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_dataloader = create_stage1_dataloader(
            config, tokenizer, split='val',
            batch_size=config['stage1']['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Training samples: {len(train_dataloader.dataset)}")
        print(f"Validation samples: {len(val_dataloader.dataset)}")
        
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        print("Falling back to direct dataset creation...")
        
        # Fallback to direct dataset creation
        train_dataset = PosterDatasetStage1(config, tokenizer, split='train')
        val_dataset = PosterDatasetStage1(config, tokenizer, split='val')
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config['stage1']['batch_size'], 
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=config['stage1']['batch_size'], 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
    
    # --- 6. Initialize Optimizer and LR Scheduler ---
    # Gather trainable parameters from the models that require gradients
    trainable_params = []
    for model_name, model in models.items():
        if model_name in ['ids_text_embedder', 'adapter']:
            trainable_params.extend(list(model.parameters()))
    
    # Filter only parameters that require gradients
    trainable_params = [p for p in trainable_params if p.requires_grad]
    print(f"Optimizer will update {len(trainable_params)} parameter groups")
    
    stage1_config = config['stage1']
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=stage1_config['learning_rate'],
        betas=(stage1_config.get('adam_beta1', 0.9), stage1_config.get('adam_beta2', 0.999)),
        eps=stage1_config.get('adam_epsilon', 1e-8),
        weight_decay=stage1_config.get('weight_decay', 0.01)
    )
    
    # Create learning rate scheduler
    lr_scheduler = create_scheduler(optimizer, config)
    
    # Create diffusion scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon"
    )
    
    print(f"Optimizer: AdamW with LR={stage1_config['learning_rate']}")
    print(f"Scheduler: {stage1_config.get('lr_scheduler_type', 'cosine')}")
    
    # --- 7. Initialize Trainer ---
    print("Initializing trainer...")
    trainer = Trainer(
        config=config,
        models=models,
        tokenizers=tokenizers,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        scheduler=scheduler
    )
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        print(f"Resuming training from: {args.resume_from_checkpoint}")
        if os.path.exists(args.resume_from_checkpoint):
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
            models['ids_text_embedder'].load_state_dict(checkpoint['ids_text_embedder'])
            models['adapter'].load_state_dict(checkpoint['adapter'])
            print("Checkpoint loaded successfully!")
        else:
            print(f"Warning: Checkpoint file not found at {args.resume_from_checkpoint}")
    
    # Start training
    print("="*60)
    print("STARTING STAGE 1 TRAINING")
    print("="*60)
    print(f"Training for {stage1_config['num_train_epochs']} epochs")
    print(f"Batch size: {stage1_config['batch_size']}")
    print(f"Gradient accumulation steps: {stage1_config['gradient_accumulation_steps']}")
    print(f"Mixed precision: {stage1_config['use_amp']}")
    print("="*60)
    
    try:
        trainer.train()
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer._save_checkpoint(trainer.current_epoch, is_best=False)
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print("Training script finished.")


if __name__ == "__main__":
    main()