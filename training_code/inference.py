"""
Inference script for PosterMaker model.
This script loads a fine-tuned model and generates posters based on text inputs.
"""

import torch
import argparse
import os
import json
import sys
from pathlib import Path
from PIL import Image
from typing import Dict, Any, List
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import load_config
from src.model_loader import load_all_models_for_stage1
from src.utils.ids_query import IDSQuery
from src.utils.ids_tokenizer import IDSTokenizer
from diffusers import DDPMScheduler
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import save_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_fine_tuned_models(config: Dict[str, Any], checkpoint_path: str, device: torch.device) -> Dict[str, torch.nn.Module]:
    """
    Load base models and apply fine-tuned weights.
    
    Args:
        config: Training configuration
        checkpoint_path: Path to fine-tuned checkpoint
        device: Device to load models on
        
    Returns:
        Dictionary of loaded models
    """
    logger.info("Loading base models...")
    models = load_all_models_for_stage1(config, device)
    
    logger.info(f"Loading fine-tuned weights from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load fine-tuned weights
    if 'ids_text_embedder' in checkpoint:
        models['ids_text_embedder'].load_state_dict(checkpoint['ids_text_embedder'])
        logger.info("✓ Loaded IDSTextEmbedder weights")
    else:
        logger.warning("IDSTextEmbedder weights not found in checkpoint")
    
    if 'adapter' in checkpoint:
        models['adapter'].load_state_dict(checkpoint['adapter'])
        logger.info("✓ Loaded Adapter weights")
    else:
        logger.warning("Adapter weights not found in checkpoint")
    
    # Set all models to evaluation mode
    for model_name, model in models.items():
        model.eval()
        logger.info(f"✓ {model_name} set to evaluation mode")
    
    return models


def create_tokenizer(config: Dict[str, Any]) -> IDSTokenizer:
    """
    Create IDS tokenizer with proper configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Configured IDSTokenizer
    """
    poster_maker_dir = Path(config['poster_maker_dir'])
    ids_database_path = poster_maker_dir / config['ids_database_path']
    
    # Initialize IDS query
    ids_query_instance = IDSQuery(ids_file_path=str(ids_database_path))
    
    # Check for vocabulary file
    vocab_file = None
    if 'ids_vocab_path' in config:
        vocab_path = poster_maker_dir / config['ids_vocab_path']
        if vocab_path.exists():
            vocab_file = str(vocab_path)
            logger.info(f"Using vocabulary file: {vocab_file}")
        else:
            logger.warning(f"Vocabulary file not found: {vocab_path}, will build from IDS database")
    
    # Create tokenizer
    tokenizer = IDSTokenizer(
        ids_query_instance=ids_query_instance,
        vocab_file=vocab_file,
        build_vocab=(vocab_file is None)
    )
    
    logger.info(f"✓ Tokenizer initialized with vocabulary size: {tokenizer.vocab_size}")
    return tokenizer


def prepare_text_inputs(text_json_path: str, tokenizer: IDSTokenizer, config: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """
    Process text inputs and create text features.
    
    Args:
        text_json_path: Path to JSON file containing text annotations
        tokenizer: IDS tokenizer
        config: Training configuration
        device: Device for tensors
        
    Returns:
        Text features tensor
    """
    logger.info(f"Loading text annotations from: {text_json_path}")
    
    if not os.path.exists(text_json_path):
        raise FileNotFoundError(f"Text annotation file not found: {text_json_path}")
    
    with open(text_json_path, 'r', encoding='utf-8') as f:
        text_data = json.load(f)
    
    # Validate text data format
    if 'texts' not in text_data or not isinstance(text_data['texts'], list):
        raise ValueError("Text JSON must contain 'texts' array")
    
    if not text_data['texts']:
        raise ValueError("No text entries found in JSON file")
    
    # Process text entries and create tokenized inputs
    all_input_ids = []
    all_attention_masks = []
    all_token_type_ids = []
    all_text_pos = []
    
    for text_entry in text_data['texts']:
        if 'content' not in text_entry or 'pos' not in text_entry:
            raise ValueError("Each text entry must have 'content' and 'pos' fields")
        
        content = text_entry['content']
        pos = text_entry['pos']  # [x1, y1, x2, y2]
        
        # Tokenize text content
        tokenized = tokenizer.encode_text(
            content,
            max_length=config['stage1']['max_seq_length'],
            add_special_tokens=True,
            use_recursive=True
        )
        
        all_input_ids.append(tokenized['input_ids'])
        all_attention_masks.append(tokenized['attention_mask'])
        all_token_type_ids.append(tokenized['token_type_ids'])
        
        # Normalize positions to [0, 1] range if needed
        normalized_pos = [
            pos[0] / 1024.0 if pos[0] > 1 else pos[0],
            pos[1] / 1024.0 if pos[1] > 1 else pos[1],
            pos[2] / 1024.0 if pos[2] > 1 else pos[2],
            pos[3] / 1024.0 if pos[3] > 1 else pos[3]
        ]
        all_text_pos.append(normalized_pos)
        
        logger.info(f"✓ Processed text: '{content}' at position {pos}")
    
    # For simplicity, use the first text entry (can be extended for multiple texts)
    input_ids = torch.tensor([all_input_ids[0]], dtype=torch.long, device=device)
    attention_mask = torch.tensor([all_attention_masks[0]], dtype=torch.long, device=device)
    token_type_ids = torch.tensor([all_token_type_ids[0]], dtype=torch.long, device=device)
    text_pos = torch.tensor([all_text_pos[0]], dtype=torch.float32, device=device)
    
    logger.info(f"✓ Text inputs prepared - shape: {input_ids.shape}")
    return input_ids, attention_mask, token_type_ids, text_pos


def create_conditioning_image(subject_image_path: str = None, subject_mask_path: str = None, 
                            text_positions: List = None, device: torch.device = None) -> torch.Tensor:
    """
    Create conditioning image for ControlNet.
    
    Args:
        subject_image_path: Optional path to subject image
        subject_mask_path: Optional path to subject mask
        text_positions: List of text positions for masking
        device: Device for tensors
        
    Returns:
        Conditioning image tensor
    """
    # For this inference script, we'll create a simple conditioning approach
    # In a full implementation, this would handle subject images and complex masking
    
    if subject_image_path and os.path.exists(subject_image_path):
        logger.info(f"Loading subject image from: {subject_image_path}")
        # Load and process subject image
        subject_image = Image.open(subject_image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        conditioning_image = transform(subject_image).unsqueeze(0).to(device)
        
        # Apply text region masking if positions provided
        if text_positions:
            for pos in text_positions:  # pos = [x1, y1, x2, y2] normalized
                x1, y1, x2, y2 = [int(p * 1024) for p in pos]
                conditioning_image[:, :, y1:y2, x1:x2] = 0.0  # Mask text regions
        
    else:
        # Create a black canvas for text-only generation
        logger.info("Creating black canvas for text generation")
        conditioning_image = torch.zeros(1, 3, 1024, 1024, device=device)
    
    logger.info(f"✓ Conditioning image prepared - shape: {conditioning_image.shape}")
    return conditioning_image


def run_diffusion_inference(models: Dict[str, torch.nn.Module], text_features: torch.Tensor, 
                          conditioning_image: torch.Tensor, device: torch.device,
                          num_inference_steps: int = 28) -> torch.Tensor:
    """
    Run the complete diffusion inference process.
    
    Args:
        models: Dictionary of loaded models
        text_features: Processed text features
        conditioning_image: ControlNet conditioning image
        device: Device for computation
        num_inference_steps: Number of denoising steps
        
    Returns:
        Generated image tensor
    """
    logger.info(f"Starting diffusion inference with {num_inference_steps} steps...")
    
    with torch.no_grad():
        # Get the correct latent shape by encoding a dummy image
        dummy_image = torch.randn(1, 3, 1024, 1024, device=device)
        sample_latents = models['vae'].encode(dummy_image).latent_dist.sample()
        scaling_factor = getattr(models['vae'].config, 'scaling_factor', 0.13025)
        sample_latents = sample_latents * scaling_factor
        
        # Initialize random noise latents
        latents = torch.randn_like(sample_latents)
        
        # Set up scheduler
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
        
        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Generating image")):
            # Expand timestep to match batch size
            timestep = t.expand(latents.shape[0])
            
            # Get ControlNet guidance
            down_block_res_samples, mid_block_res_sample = models['text_render_net'](
                sample=latents,
                timestep=timestep,
                encoder_hidden_states=text_features,
                controlnet_cond=conditioning_image,
                return_dict=False
            )
            
            # Predict noise with main transformer
            noise_pred = models['transformer'](
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=text_features,
                pooled_projections=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False
            )[0]
            
            # Update latents using scheduler
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Decode latents to image
        logger.info("Decoding latents to image...")
        latents = latents / scaling_factor
        image = models['vae'].decode(latents).sample
        
        # Post-process image
        image = (image / 2 + 0.5).clamp(0, 1)
        
        logger.info("✓ Image generation completed")
        return image


def run_inference(args):
    """
    Main function to run the inference process.
    
    Args:
        args: Command line arguments
    """
    try:
        # Setup
        logger.info("="*60)
        logger.info("POSTERMAKER INFERENCE")
        logger.info("="*60)
        
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        if device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        
        # Load models and fine-tuned weights
        models = load_fine_tuned_models(config, args.checkpoint_path, device)
        
        # Create tokenizer
        tokenizer = create_tokenizer(config)
        
        # Prepare text inputs
        input_ids, attention_mask, token_type_ids, text_pos = prepare_text_inputs(
            args.text_json_path, tokenizer, config, device
        )
        
        # Process text through IDS embedder and adapter
        logger.info("Processing text through trained models...")
        with torch.no_grad():
            text_embeds = models['ids_text_embedder'].forward_tokenized_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                text_pos=text_pos
            )
            text_features = models['adapter'](text_embeds)
            logger.info(f"✓ Text features shape: {text_features.shape}")
        
        # Create conditioning image
        text_positions = None
        if args.text_json_path:
            with open(args.text_json_path, 'r', encoding='utf-8') as f:
                text_data = json.load(f)
                text_positions = [
                    [pos / 1024.0 if pos > 1 else pos for pos in text['pos']]
                    for text in text_data['texts']
                ]
        
        conditioning_image = create_conditioning_image(
            subject_image_path=getattr(args, 'subject_image', None),
            subject_mask_path=getattr(args, 'subject_mask', None),
            text_positions=text_positions,
            device=device
        )
        
        # Run diffusion inference
        generated_image = run_diffusion_inference(
            models=models,
            text_features=text_features,
            conditioning_image=conditioning_image,
            device=device,
            num_inference_steps=args.num_steps
        )
        
        # Save output
        logger.info(f"Saving generated image to: {args.output_path}")
        os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
        
        save_image(generated_image, args.output_path, normalize=False)
        
        # Also save as PIL image for better compatibility
        pil_output_path = args.output_path.replace('.png', '_pil.png')
        pil_image = transforms.ToPILImage()(generated_image.squeeze(0))
        pil_image.save(pil_output_path)
        
        logger.info("="*60)
        logger.info("INFERENCE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Generated image saved to: {args.output_path}")
        logger.info(f"PIL version saved to: {pil_output_path}")
        logger.info(f"Image size: {generated_image.shape}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run inference with a trained PosterMaker model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference with text JSON
  python inference.py --config configs/train_config.yaml --checkpoint_path training_output/checkpoints/best_model.pth --text_json_path sample_text.json --output_path generated_poster.png
  
  # With custom subject image
  python inference.py --config configs/train_config.yaml --checkpoint_path training_output/checkpoints/best_model.pth --text_json_path sample_text.json --subject_image subject.jpg --output_path poster_with_subject.png
        """
    )
    
    parser.add_argument("--config", type=str, required=True,
                       help="Path to the training config file")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to the trained .pth checkpoint file (e.g., best_model.pth)")
    parser.add_argument("--text_json_path", type=str, required=True,
                       help="Path to JSON file with text content and positions")
    parser.add_argument("--output_path", type=str, default="inference_output.png",
                       help="Path to save the generated image")
    parser.add_argument("--subject_image", type=str, default=None,
                       help="Optional path to subject image for poster background")
    parser.add_argument("--subject_mask", type=str, default=None,
                       help="Optional path to subject mask")
    parser.add_argument("--num_steps", type=int, default=28,
                       help="Number of diffusion inference steps (default: 28)")
    
    args = parser.parse_args()
    
    # Validate required files exist
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1
    
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found: {args.checkpoint_path}")
        return 1
    
    if not os.path.exists(args.text_json_path):
        print(f"Error: Text JSON file not found: {args.text_json_path}")
        return 1
    
    return run_inference(args)


if __name__ == "__main__":
    sys.exit(main())