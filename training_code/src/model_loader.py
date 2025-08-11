"""
Model loading and weight management for PosterMaker IDS-based training pipeline.
This module handles the complex logic of loading all pre-trained model components,
initializing new modules, and applying weight freezing for Stage 1 training.
"""

import torch
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from diffusers import AutoencoderKL, SD3Transformer2DModel

# Import our custom model classes
from .models.controlnet_sd3 import SD3ControlNetModel
from .models.adapter_models import LinearAdapterWithLayerNorm
from .models.ids_text_embedder import IDSTextEmbedder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_all_models_for_stage1(config: Dict[str, Any], device: torch.device) -> Dict[str, torch.nn.Module]:
    """
    Load all necessary models for Stage 1 training, initialize new components,
    load pre-trained weights, and apply the correct weight freezing.

    Args:
        config (Dict[str, Any]): The loaded training configuration
        device (torch.device): The device to move models to (e.g., 'cuda')

    Returns:
        Dict[str, torch.nn.Module]: A dictionary containing all model components
    """
    logger.info("Loading all models for Stage 1 training...")
    
    poster_maker_dir = Path(config['poster_maker_dir'])
    
    # --- 1. Load Pre-trained SD3 Components ---
    logger.info("Loading pre-trained SD3 components...")
    
    sd3_path = poster_maker_dir / config['sd3_model_path']
    
    # Load VAE (Vector Auto-Encoder for latent space conversion)
    vae_path = sd3_path / 'vae'
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE path not found: {vae_path}")
    
    vae = AutoencoderKL.from_pretrained(str(vae_path))
    logger.info(f"Loaded VAE from {vae_path}")
    
    # Load Transformer (SD3 main diffusion model)
    transformer_path = sd3_path / 'transformer'
    if not transformer_path.exists():
        raise FileNotFoundError(f"Transformer path not found: {transformer_path}")
    
    transformer = SD3Transformer2DModel.from_pretrained(str(transformer_path))
    logger.info(f"Loaded SD3 Transformer from {transformer_path}")
    
    # Note: Text encoders are not needed for Stage 1 training, saving VRAM
    
    # --- 2. Load Pre-trained ControlNets ---
    logger.info("Loading pre-trained ControlNets...")
    
    # Load TextRenderNet
    text_render_net_path = poster_maker_dir / config['original_textrender_net_path']
    if not text_render_net_path.exists():
        raise FileNotFoundError(f"TextRenderNet weights not found: {text_render_net_path}")
    
    # Create TextRenderNet from transformer config
    text_render_net = SD3ControlNetModel.from_transformer(
        transformer, 
        num_layers=transformer.config.num_layers,
        # additional_in_channel=1,  # For inpainting mask
        load_weights_from_transformer=True
    )
    
    # Load TextRenderNet weights (includes both controlnet and adapter weights)
    textrender_net_state_dict = torch.load(text_render_net_path, map_location='cpu')
    
    # Extract TextRenderNet weights based on state dict structure
    if 'controlnet_text' in textrender_net_state_dict:
        text_render_net.load_state_dict(textrender_net_state_dict['controlnet_text'], strict=False)
        logger.info(f"✓ Loaded TextRenderNet from 'controlnet_text' key in {text_render_net_path}")
    else:
        # Fallback: assume the entire state dict is for the controlnet
        text_render_net.load_state_dict(textrender_net_state_dict, strict=False)
        logger.info(f"✓ Loaded TextRenderNet (direct mode) from {text_render_net_path}")
        logger.warning("Using fallback loading mode - check if this is expected")
    
    # --- 3. Initialize Our New/Trainable Modules ---
    logger.info("Initializing new trainable modules...")
    
    # Initialize the new IDSTextEmbedder
    ids_database_path = poster_maker_dir / config['ids_database_path']
    if not ids_database_path.exists():
        raise FileNotFoundError(f"IDS database not found: {ids_database_path}")
    
    # Check for vocabulary file
    vocab_file = None
    if 'ids_vocab_path' in config:
        vocab_path = poster_maker_dir / config['ids_vocab_path']
        if vocab_path.exists():
            vocab_file = str(vocab_path)
            logger.info(f"Using existing vocabulary: {vocab_file}")
        else:
            logger.info("Vocabulary file not found, will build from scratch")
    
    # Check for pre-trained character features
    char2feat_path = None
    if 'char2feat_path' in config:
        char2feat_full_path = poster_maker_dir / config['char2feat_path']
        if char2feat_full_path.exists():
            char2feat_path = str(char2feat_full_path)
            logger.info(f"Using pre-trained character features: {char2feat_path}")
        else:
            logger.warning(f"Character features not found: {char2feat_full_path}")
    
    # Initialize IDSTextEmbedder
    ids_text_embedder = IDSTextEmbedder(
        ids_database_path=str(ids_database_path),
        vocab_file=vocab_file,
        ids_embed_dim=config['stage1']['embedding_dim'],
        max_seq_length=config['stage1']['max_seq_length'],
        char2feat_path=char2feat_path,
        char2feat_alignment_weight=0.3  # Warm start with pre-trained features
    )
    logger.info(f"Initialized IDSTextEmbedder with vocab size: {ids_text_embedder.tokenizer.vocab_size}")
    
    # Initialize the Adapter
    # The adapter transforms IDS embedder output (128D) to ControlNet input dimension
    hidden_dim = 128  # Output dimension from IDSTextEmbedder (64 + 32 + 32)
    projection_dim = transformer.config.joint_attention_dim  # SD3 expects 4096D features
    
    adapter = LinearAdapterWithLayerNorm(
        hidden_dim=hidden_dim, 
        projection_dim=projection_dim
    )
    
    # Load pre-trained adapter weights from the TextRenderNet state dict
    if 'adapter' in textrender_net_state_dict:
        try:
            adapter.load_state_dict(textrender_net_state_dict['adapter'], strict=False)
            logger.info("✓ Loaded pre-trained adapter weights from TextRenderNet state dict")
        except Exception as e:
            logger.warning(f"Could not load adapter weights from TextRenderNet: {e}. Using Xavier initialization.")
            adapter.apply(lambda m: torch.nn.init.xavier_uniform_(m.weight) if hasattr(m, 'weight') else None)
    else:
        logger.warning("Adapter weights not found in TextRenderNet state dict, using Xavier initialization")
        adapter.apply(lambda m: torch.nn.init.xavier_uniform_(m.weight) if hasattr(m, 'weight') else None)
    
    logger.info(f"Initialized Adapter: {hidden_dim}D -> {projection_dim}D")
    
    # --- 4. Apply Weight Freezing for Stage 1 ---
    logger.info("Applying weight freezing for Stage 1 training...")
    
    # Freeze all pre-trained models (no gradients needed)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    text_render_net.requires_grad_(False)
    
    # Enable training for our target modules
    ids_text_embedder.requires_grad_(True)
    adapter.requires_grad_(True)
    
    logger.info("Weight freezing applied:")
    logger.info("  Frozen: VAE, SD3 Transformer, TextRenderNet")
    logger.info("  Trainable: IDSTextEmbedder, Adapter")
    
    # --- 5. Move to Device and Set Training Modes ---
    logger.info(f"Moving models to device: {device}")
    
    models = {
        "vae": vae.to(device),
        "transformer": transformer.to(device), 
        "text_render_net": text_render_net.to(device),
        "ids_text_embedder": ids_text_embedder.to(device),
        "adapter": adapter.to(device)
    }
    
    # Set appropriate training modes
    models['vae'].eval()  # Always in eval mode
    models['transformer'].eval()  # Frozen, eval mode
    models['text_render_net'].eval()  # Frozen, eval mode
    models['ids_text_embedder'].train()  # Trainable, train mode
    models['adapter'].train()  # Trainable, train mode
    
    # Log trainable parameters
    total_params = 0
    trainable_params = 0
    
    for name, model in models.items():
        model_params = sum(p.numel() for p in model.parameters())
        model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params += model_params
        trainable_params += model_trainable
        
        logger.info(f"  {name}: {model_params:,} total, {model_trainable:,} trainable")
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    logger.info("All models loaded successfully for Stage 1 training!")
    
    return models


def save_model_checkpoint(models: Dict[str, torch.nn.Module], 
                         config: Dict[str, Any], 
                         epoch: int, 
                         step: int,
                         optimizer_state: Optional[Dict] = None,
                         scheduler_state: Optional[Dict] = None) -> str:
    """
    Save a checkpoint containing only the trainable model states.
    
    Args:
        models: Dictionary of all models
        config: Training configuration
        epoch: Current epoch
        step: Current step
        optimizer_state: Optimizer state dict (optional)
        scheduler_state: Scheduler state dict (optional)
        
    Returns:
        str: Path to saved checkpoint
    """
    checkpoint_dir = Path(config.get('output_dir', './training_output')) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"stage1_checkpoint_epoch_{epoch}_step_{step}.pth"
    
    # Only save trainable model states
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'config': config,
        'model_states': {
            'ids_text_embedder': models['ids_text_embedder'].state_dict(),
            'adapter': models['adapter'].state_dict()
        }
    }
    
    if optimizer_state is not None:
        checkpoint['optimizer_state'] = optimizer_state
    
    if scheduler_state is not None:
        checkpoint['scheduler_state'] = scheduler_state
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    return str(checkpoint_path)


def load_model_checkpoint(models: Dict[str, torch.nn.Module], 
                         checkpoint_path: str,
                         device: torch.device,
                         load_optimizer: bool = True) -> Dict[str, Any]:
    """
    Load a checkpoint and restore trainable model states.
    
    Args:
        models: Dictionary of all models
        checkpoint_path: Path to checkpoint file
        device: Device to load to
        load_optimizer: Whether to load optimizer/scheduler states
        
    Returns:
        Dict containing checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model states
    models['ids_text_embedder'].load_state_dict(checkpoint['model_states']['ids_text_embedder'])
    models['adapter'].load_state_dict(checkpoint['model_states']['adapter'])
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"  Epoch: {checkpoint['epoch']}, Step: {checkpoint['step']}")
    
    return checkpoint


def count_trainable_parameters(models: Dict[str, torch.nn.Module]) -> Dict[str, int]:
    """
    Count trainable parameters for each model.
    
    Args:
        models: Dictionary of models
        
    Returns:
        Dictionary with parameter counts for each model
    """
    param_counts = {}
    
    for name, model in models.items():
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        param_counts[name] = {
            'trainable': trainable_params,
            'total': total_params,
            'percentage': trainable_params / total_params * 100 if total_params > 0 else 0
        }
    
    return param_counts


def verify_model_setup(models: Dict[str, torch.nn.Module]) -> bool:
    """
    Verify that models are correctly set up for training.
    
    Args:
        models: Dictionary of models
        
    Returns:
        bool: True if setup is correct
    """
    # Check that frozen models are in eval mode and have no gradients
    frozen_models = ['vae', 'transformer', 'text_render_net']
    trainable_models = ['ids_text_embedder', 'adapter']
    
    for name in frozen_models:
        if name in models:
            model = models[name]
            if model.training:
                logger.error(f"Frozen model {name} should be in eval mode")
                return False
            
            if any(p.requires_grad for p in model.parameters()):
                logger.error(f"Frozen model {name} should have no trainable parameters")
                return False
    
    # Check that trainable models are in train mode and have gradients
    for name in trainable_models:
        if name in models:
            model = models[name]
            if not model.training:
                logger.warning(f"Trainable model {name} should be in train mode")
            
            if not any(p.requires_grad for p in model.parameters()):
                logger.error(f"Trainable model {name} should have trainable parameters")
                return False
    
    logger.info("Model setup verification passed!")
    return True