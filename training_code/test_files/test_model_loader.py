"""
Test script for validating the model loading functionality.
This script tests the load_all_models_for_stage1 function and verifies proper setup.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from src.config import load_config
from src.model_loader import (
    load_all_models_for_stage1, 
    verify_model_setup, 
    count_trainable_parameters,
    save_model_checkpoint
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_loading(config: dict, device: torch.device):
    """Test the model loading functionality"""
    logger.info("Testing model loading...")
    
    try:
        # Load all models
        models = load_all_models_for_stage1(config, device)
        
        logger.info("Model loading test passed!")
        logger.info(f"Loaded models: {list(models.keys())}")
        
        # Verify model setup
        if verify_model_setup(models):
            logger.info("Model setup verification passed!")
        else:
            logger.error("Model setup verification failed!")
            return False
        
        # Count parameters
        param_counts = count_trainable_parameters(models)
        logger.info("Parameter counts:")
        for name, counts in param_counts.items():
            logger.info(f"  {name}: {counts['trainable']:,} trainable / {counts['total']:,} total ({counts['percentage']:.1f}%)")
        
        return models
        
    except Exception as e:
        logger.error(f"Model loading test failed: {e}")
        raise


def test_model_forward_pass(models: dict, config: dict, device: torch.device):
    """Test basic forward pass through models"""
    logger.info("Testing model forward passes...")
    
    try:
        # Test IDSTextEmbedder
        ids_embedder = models['ids_text_embedder']
        
        # Create sample text data
        sample_texts = [
            {'content': '你好世界', 'pos': [100, 100, 200, 150]},
            {'content': '测试文本', 'pos': [300, 200, 400, 250]}
        ]
        
        # Forward pass through IDS embedder
        with torch.no_grad():
            text_features = ids_embedder(sample_texts)  # Should be (112, 128)
            
        logger.info(f"IDSTextEmbedder output shape: {text_features.shape}")
        
        if text_features.shape != torch.Size([112, 128]):
            logger.error(f"Expected shape (112, 128), got {text_features.shape}")
            return False
        
        # Test Adapter
        adapter = models['adapter']
        
        # Test adapter with sample input
        sample_input = torch.randn(16, 128, device=device)  # Batch of features
        
        with torch.no_grad():
            adapter_output = adapter(sample_input)
        
        logger.info(f"Adapter output shape: {adapter_output.shape}")
        
        expected_dim = 4096  # SD3 joint_attention_dim
        if adapter_output.shape[-1] != expected_dim:
            logger.error(f"Expected last dimension {expected_dim}, got {adapter_output.shape[-1]}")
            return False
        
        # Test VAE encoding/decoding (basic shape test)
        vae = models['vae']
        
        # Create sample image batch
        sample_images = torch.randn(1, 3, 1024, 1024, device=device)
        
        with torch.no_grad():
            # Encode to latents
            latents = vae.encode(sample_images).latent_dist.sample()
            logger.info(f"VAE latents shape: {latents.shape}")
            
            # Decode back
            decoded = vae.decode(latents).sample
            logger.info(f"VAE decoded shape: {decoded.shape}")
        
        logger.info("Model forward pass tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Model forward pass test failed: {e}")
        return False


def test_checkpoint_save_load(models: dict, config: dict):
    """Test checkpoint saving and loading"""
    logger.info("Testing checkpoint save/load...")
    
    try:
        # Save a test checkpoint
        checkpoint_path = save_model_checkpoint(
            models=models,
            config=config,
            epoch=1,
            step=100,
            optimizer_state={'test': 'optimizer_state'},
            scheduler_state={'test': 'scheduler_state'}
        )
        
        # Verify checkpoint file exists
        if not Path(checkpoint_path).exists():
            logger.error(f"Checkpoint file not created: {checkpoint_path}")
            return False
        
        # Test loading checkpoint
        from src.model_loader import load_model_checkpoint
        
        checkpoint_data = load_model_checkpoint(
            models=models,
            checkpoint_path=checkpoint_path,
            device=next(models['ids_text_embedder'].parameters()).device
        )
        
        # Verify checkpoint data
        assert checkpoint_data['epoch'] == 1
        assert checkpoint_data['step'] == 100
        assert 'optimizer_state' in checkpoint_data
        assert 'scheduler_state' in checkpoint_data
        
        logger.info("Checkpoint save/load test passed!")
        
        # Clean up test checkpoint
        Path(checkpoint_path).unlink()
        
        return True
        
    except Exception as e:
        logger.error(f"Checkpoint save/load test failed: {e}")
        return False


def test_device_transfer(models: dict, available_devices: list):
    """Test transferring models between devices"""
    logger.info("Testing device transfer...")
    
    if len(available_devices) < 2:
        logger.info("Only one device available, skipping device transfer test")
        return True
    
    try:
        # Test moving to different device
        target_device = available_devices[1]  # Try second device
        
        # Move one model to test
        ids_embedder = models['ids_text_embedder']
        original_device = next(ids_embedder.parameters()).device
        
        ids_embedder.to(target_device)
        new_device = next(ids_embedder.parameters()).device
        
        if new_device != target_device:
            logger.error(f"Device transfer failed: expected {target_device}, got {new_device}")
            return False
        
        # Move back
        ids_embedder.to(original_device)
        
        logger.info("Device transfer test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Device transfer test failed: {e}")
        return False


def main():
    """Main function to run all tests"""
    parser = argparse.ArgumentParser(description="Test PosterMaker model loading functionality")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to the training configuration file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--test-loading', action='store_true',
                       help='Test model loading')
    parser.add_argument('--test-forward', action='store_true',
                       help='Test model forward passes')
    parser.add_argument('--test-checkpoint', action='store_true',
                       help='Test checkpoint save/load')
    parser.add_argument('--test-device', action='store_true',
                       help='Test device transfer')
    parser.add_argument('--test-all', action='store_true',
                       help='Run all tests')
    
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
    
    # Get available devices for testing
    available_devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            available_devices.append(torch.device(f'cuda:{i}'))
    
    logger.info(f"Available devices: {available_devices}")
    
    models = None
    
    # Run tests based on arguments
    if args.test_all or args.test_loading:
        models = test_model_loading(config, device)
        if models is None:
            logger.error("Model loading failed, stopping tests")
            return 1
    
    if args.test_all or args.test_forward:
        if models is None:
            models = test_model_loading(config, device)
        if not test_model_forward_pass(models, config, device):
            logger.error("Forward pass tests failed")
            return 1
    
    if args.test_all or args.test_checkpoint:
        if models is None:
            models = test_model_loading(config, device)
        if not test_checkpoint_save_load(models, config):
            logger.error("Checkpoint tests failed")
            return 1
    
    if args.test_all or args.test_device:
        if models is None:
            models = test_model_loading(config, device)
        if not test_device_transfer(models, available_devices):
            logger.error("Device transfer tests failed")
            return 1
    
    logger.info("All tests completed successfully! 🎉")
    return 0


if __name__ == "__main__":
    sys.exit(main())