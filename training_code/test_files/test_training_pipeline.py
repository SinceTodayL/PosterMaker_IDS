"""
Comprehensive test script for the complete PosterMaker training pipeline.
This script tests the integration of all components in the training process.
"""

import os
import sys
import argparse
import logging
import tempfile
import shutil
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import DDPMScheduler

from src.config import load_config
from src.model_loader import load_all_models_for_stage1
from src.utils.ids_query import IDSQuery
from src.utils.ids_tokenizer import IDSTokenizer
from src.trainer import Trainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_dataset_structure(temp_dir: str, config: dict):
    """Create a minimal dummy dataset structure for testing."""
    dataset_dir = os.path.join(temp_dir, 'dummy_dataset')
    
    # Create train and val directories
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Create dummy samples
    for split_dir, num_samples in [(train_dir, 5), (val_dir, 2)]:
        for i in range(num_samples):
            sample_dir = os.path.join(split_dir, f'sample_{i:03d}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # Create dummy image files (1x1 RGB images)
            import PIL.Image
            dummy_image = PIL.Image.new('RGB', (1024, 1024), color='white')
            dummy_image.save(os.path.join(sample_dir, 'image.png'))
            
            dummy_mask = PIL.Image.new('L', (1024, 1024), color='black')
            dummy_mask.save(os.path.join(sample_dir, 'subject_mask.png'))
            
            # Create dummy annotation
            import json
            annotation = {
                "prompt": "A poster with text",
                "texts": [
                    {
                        "content": "测试文本",
                        "pos": [100, 100, 300, 150]
                    }
                ]
            }
            
            with open(os.path.join(sample_dir, 'annotation.json'), 'w', encoding='utf-8') as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)
    
    # Update config to use dummy dataset
    config['dataset_dir'] = dataset_dir
    logger.info(f"Created dummy dataset at: {dataset_dir}")
    
    return dataset_dir


def create_dummy_dataloader(tokenizer, config, batch_size=2, num_samples=4):
    """Create a dummy dataloader for testing."""
    from torch.utils.data import Dataset, DataLoader
    
    class MinimalDummyDataset(Dataset):
        def __init__(self, num_samples, tokenizer, max_seq_length):
            self.num_samples = num_samples
            self.tokenizer = tokenizer
            self.max_seq_length = max_seq_length
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Create dummy data that matches our expected format
            dummy_encoding = self.tokenizer.encode_text(
                "测试文本", 
                max_length=self.max_seq_length,
                add_special_tokens=True,
                use_recursive=False,  # Use simple decomposition for training
                enhance_rare_chars=False
            )
            
            return {
                "pixel_values": torch.randn(3, 1024, 1024),
                "conditioning_pixel_values": torch.randn(3, 1024, 1024),
                "conditioning_mask": torch.randn(1, 1024, 1024),
                "input_ids": torch.tensor(dummy_encoding['input_ids'], dtype=torch.long),
                "attention_mask": torch.tensor(dummy_encoding['attention_mask'], dtype=torch.long),
                "token_type_ids": torch.tensor(dummy_encoding['token_type_ids'], dtype=torch.long),
                "text_pos": torch.tensor([0.1, 0.1, 0.9, 0.9], dtype=torch.float32),
                "text_content": "测试文本",
                "sample_id": f"dummy_sample_{idx}"
            }
    
    def collate_fn(batch):
        # Stack tensors properly
        return {
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
            "conditioning_pixel_values": torch.stack([item["conditioning_pixel_values"] for item in batch]),
            "conditioning_mask": torch.stack([item["conditioning_mask"] for item in batch]),
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "token_type_ids": torch.stack([item["token_type_ids"] for item in batch]),
            "text_pos": torch.stack([item["text_pos"] for item in batch]),
            "text_content": [item["text_content"] for item in batch],
            "sample_ids": [item["sample_id"] for item in batch]
        }
    
    dataset = MinimalDummyDataset(num_samples, tokenizer, config['stage1']['max_seq_length'])
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    return dataloader


def test_complete_training_integration(config, device, temp_dir):
    """Test the complete training integration."""
    logger.info("Testing complete training integration...")
    
    try:
        # Create temporary output directory
        config['output_dir'] = os.path.join(temp_dir, 'training_output')
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # Load models
        models = load_all_models_for_stage1(config, device)
        
        # Create tokenizer with updated interface
        # Get paths based on updated configuration
        if config['ids_database_path'].startswith('./'):
            # IDS database path is relative to training_code
            ids_database_path = Path(config['ids_database_path'])
        else:
            # IDS database path is relative to poster_maker_dir (legacy support)
            poster_maker_dir = Path(config['poster_maker_dir'])
            ids_database_path = poster_maker_dir / config['ids_database_path']
        
        vocab_file = None
        if 'ids_vocab_path' in config:
            if config['ids_vocab_path'].startswith('./'):
                # Vocabulary path is relative to training_code
                vocab_path = Path(config['ids_vocab_path'])
            else:
                # Vocabulary path is relative to poster_maker_dir (legacy support)
                poster_maker_dir = Path(config['poster_maker_dir'])
                vocab_path = poster_maker_dir / config['ids_vocab_path']
                
            if vocab_path.exists():
                vocab_file = str(vocab_path)
        
        tokenizer = IDSTokenizer(
            ids_database_path=str(ids_database_path),
            vocab_file=vocab_file,
            preserve_rare_chars=True
        )
        
        # Create dummy dataloaders
        train_dataloader = create_dummy_dataloader(tokenizer, config, batch_size=2, num_samples=6)
        val_dataloader = create_dummy_dataloader(tokenizer, config, batch_size=2, num_samples=4)
        
        # Create optimizer and scheduler
        trainable_params = []
        trainable_params.extend(models['ids_text_embedder'].parameters())
        trainable_params.extend(models['adapter'].parameters())
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=config['stage1']['learning_rate'],
            weight_decay=0.01
        )
        
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['stage1']['num_train_epochs'],
            eta_min=1e-6
        )
        
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
        
        # Initialize trainer
        trainer = Trainer(
            config=config,
            models=models,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            scheduler=scheduler
        )
        
        # Test a few training steps
        logger.info("Testing training steps...")
        
        # Backup original config values
        original_epochs = config['stage1']['num_train_epochs']
        original_val_steps = config['stage1']['validation_steps']
        
        # Temporarily modify for testing
        config['stage1']['num_train_epochs'] = 2
        config['stage1']['validation_steps'] = 5  # Validate after 5 steps
        
        # Run a short training session
        trainer.train()
        
        # Restore original config
        config['stage1']['num_train_epochs'] = original_epochs
        config['stage1']['validation_steps'] = original_val_steps
        
        # Check if checkpoints were created
        checkpoint_dir = os.path.join(config['output_dir'], 'checkpoints')
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        
        logger.info(f"Checkpoints created: {checkpoints}")
        
        # Test checkpoint loading
        if checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
            checkpoint = torch.load(checkpoint_path, map_location=device)
            logger.info(f"✓ Checkpoint loaded successfully. Keys: {list(checkpoint.keys())}")
        
        logger.info("✓ Complete training integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Complete training integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_pipeline_components(config, device):
    """Test individual training pipeline components."""
    logger.info("Testing training pipeline components...")
    
    try:
        # Test 1: Config loading
        assert 'stage1' in config, "Config must contain stage1 section"
        assert 'poster_maker_dir' in config, "Config must contain poster_maker_dir"
        logger.info("✓ Configuration structure valid")
        
        # Test 2: Model loading
        models = load_all_models_for_stage1(config, device)
        required_models = ['vae', 'transformer', 'text_render_net', 'ids_text_embedder', 'adapter']
        for model_name in required_models:
            assert model_name in models, f"Model {model_name} not found"
        logger.info("✓ All required models loaded")
        
        # Test 3: Tokenizer initialization
        # Get paths based on updated configuration
        if config['ids_database_path'].startswith('./'):
            # IDS database path is relative to training_code
            ids_database_path = Path(config['ids_database_path'])
        else:
            # IDS database path is relative to poster_maker_dir (legacy support)
            poster_maker_dir = Path(config['poster_maker_dir'])
            ids_database_path = poster_maker_dir / config['ids_database_path']
        
        vocab_file = None
        if 'ids_vocab_path' in config:
            if config['ids_vocab_path'].startswith('./'):
                # Vocabulary path is relative to training_code
                vocab_path = Path(config['ids_vocab_path'])
            else:
                # Vocabulary path is relative to poster_maker_dir (legacy support)
                poster_maker_dir = Path(config['poster_maker_dir'])
                vocab_path = poster_maker_dir / config['ids_vocab_path']
                
            if vocab_path.exists():
                vocab_file = str(vocab_path)
        
        tokenizer = IDSTokenizer(
            ids_database_path=str(ids_database_path),
            vocab_file=vocab_file,
            preserve_rare_chars=True
        )
        
        assert tokenizer.vocab_size > 0, "Tokenizer must have non-zero vocabulary"
        logger.info(f"✓ Tokenizer initialized with vocab size: {tokenizer.vocab_size}")
        
        # Test 4: Data loading
        train_dataloader = create_dummy_dataloader(tokenizer, config, batch_size=2, num_samples=4)
        val_dataloader = create_dummy_dataloader(tokenizer, config, batch_size=2, num_samples=2)
        
        # Test batch structure
        train_batch = next(iter(train_dataloader))
        required_keys = ['pixel_values', 'conditioning_pixel_values', 'input_ids', 
                        'attention_mask', 'token_type_ids', 'text_pos']
        for key in required_keys:
            assert key in train_batch, f"Batch must contain {key}"
        logger.info("✓ Data loading and batch structure valid")
        
        # Test 5: Optimizer setup
        trainable_params = []
        trainable_params.extend(models['ids_text_embedder'].parameters())
        trainable_params.extend(models['adapter'].parameters())
        
        optimizer = optim.AdamW(trainable_params, lr=config['stage1']['learning_rate'])
        assert len(optimizer.param_groups) > 0, "Optimizer must have parameter groups"
        logger.info("✓ Optimizer setup successful")
        
        logger.info("All training pipeline components tested successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Training pipeline component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run training pipeline tests."""
    parser = argparse.ArgumentParser(description="Test PosterMaker training pipeline")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to the training configuration file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--test-components', action='store_true',
                       help='Test individual pipeline components')
    parser.add_argument('--test-integration', action='store_true',
                       help='Test complete training integration')
    parser.add_argument('--test-all', action='store_true',
                       help='Run all pipeline tests')
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    success = True
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temporary directory: {temp_dir}")
        
        # Run tests based on arguments
        if args.test_all or args.test_components:
            logger.info("="*60)
            logger.info("TESTING TRAINING PIPELINE COMPONENTS")
            logger.info("="*60)
            if not test_training_pipeline_components(config, device):
                success = False
        
        if args.test_all or args.test_integration:
            logger.info("="*60)
            logger.info("TESTING COMPLETE TRAINING INTEGRATION")
            logger.info("="*60)
            if not test_complete_training_integration(config, device, temp_dir):
                success = False
    
    if success:
        logger.info("🎉 All training pipeline tests completed successfully!")
        return 0
    else:
        logger.error("❌ Some training pipeline tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())