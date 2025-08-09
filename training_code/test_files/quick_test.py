"""
Quick test to verify the forward pass implementation works correctly.
This is a minimal test that can be run without a full dataset.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_forward_pass_test():
    """Quick test of forward pass implementation."""
    config_path = "configs/train_config.yaml"
    
    try:
        # Test config loading
        config = load_config(config_path)
        logger.info("✓ Configuration loaded successfully")
        
        # Test device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"✓ Using device: {device}")
        
        # Test tensor operations
        dummy_tensor = torch.randn(2, 3, 1024, 1024, device=device)
        logger.info(f"✓ Tensor creation successful: {dummy_tensor.shape}")
        
        # Test imports
        from src.models.ids_text_embedder import IDSTextEmbedder
        from src.models.adapter_models import LinearAdapterWithLayerNorm
        from src.trainer import Trainer
        logger.info("✓ All modules imported successfully")
        
        # Test basic functionality
        adapter = LinearAdapterWithLayerNorm(128, 4096).to(device)
        dummy_input = torch.randn(2, 128, device=device)
        output = adapter(dummy_input)
        logger.info(f"✓ Adapter test successful: {dummy_input.shape} -> {output.shape}")
        
        logger.info("🎉 Quick test passed! Ready for full testing.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_forward_pass_test()
    sys.exit(0 if success else 1)