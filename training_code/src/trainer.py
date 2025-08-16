"""
Training engine for PosterMaker Stage 1 IDS-based fine-tuning.
This module provides the main Trainer class that handles the training loop,
validation, checkpointing, and progress monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
import os
import logging
import time
from pathlib import Path
from tqdm import tqdm
from tqdm.auto import tqdm as auto_tqdm
import json
from torchvision.utils import save_image
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from .model_loader import save_model_checkpoint, count_trainable_parameters

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    The main training engine for the PosterMaker Stage 1 fine-tuning.
    This class handles the training loop, validation, checkpointing, and progress monitoring.
    
    Stage 1 focuses on training the IDSTextEmbedder and Adapter modules while keeping
    the pre-trained SD3 components frozen.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 models: Dict[str, torch.nn.Module],
                 tokenizers: Dict[str, Any],
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler,  # Learning rate scheduler
                 device: torch.device,
                 scheduler  # Diffusers noise scheduler
                 ):
        """
        Initialize the Trainer with all necessary components.
        
        Args:
            config: Training configuration dictionary
            models: Dictionary containing all model components
            tokenizers: Dictionary containing tokenizers for text processing
            train_dataloader: Training data loader
            val_dataloader: Validation data loader  
            optimizer: Optimizer for trainable parameters
            lr_scheduler: Learning rate scheduler
            device: Training device (cuda/cpu)
            scheduler: Diffusers noise scheduler for diffusion process
        """
        self.config = config
        self.models = models
        self.tokenizers = tokenizers
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.scheduler = scheduler
        
        # Unpack text encoders and tokenizers for convenience
        self.text_encoder = self.models['text_encoder']
        self.text_encoder_2 = self.models['text_encoder_2']
        self.text_encoder_3 = self.models['text_encoder_3']
        self.tokenizer = self.tokenizers['tokenizer']
        self.tokenizer_2 = self.tokenizers['tokenizer_2']
        self.tokenizer_3 = self.tokenizers['tokenizer_3']
        self.tokenizer_max_length = self.tokenizer.model_max_length

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Stage 1 specific configuration
        self.stage1_config = config['stage1']
        self.num_epochs = self.stage1_config['num_train_epochs']
        self.gradient_accumulation_steps = self.stage1_config['gradient_accumulation_steps']
        self.save_steps = self.stage1_config['save_steps']
        self.validation_steps = self.stage1_config['validation_steps']
        self.use_amp = self.stage1_config['use_amp']
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Output directories
        self.output_dir = Path(config.get('output_dir', './training_output'))
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        
        # Create output directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Setup logging
        self._setup_logging()
        
        # Log initialization info
        self._log_initialization_info()
        
        logger.info("Trainer initialized successfully for Stage 1 training")
    
    def _setup_logging(self):
        """Setup file logging for training progress."""
        log_file = self.log_dir / f'training_{int(time.time())}.log'
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        logger.info(f"Training logs will be saved to: {log_file}")
    
    def _log_initialization_info(self):
        """Log important information about the training setup."""
        # Log model parameter counts
        param_counts = count_trainable_parameters(self.models)
        
        logger.info("=== Training Setup Information ===")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed Precision: {self.use_amp}")
        logger.info(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        logger.info(f"Number of Epochs: {self.num_epochs}")
        logger.info(f"Training Samples: {len(self.train_dataloader.dataset)}")
        logger.info(f"Validation Samples: {len(self.val_dataloader.dataset)}")
        
        logger.info("Model Parameter Counts:")
        total_trainable = 0
        total_params = 0
        for name, counts in param_counts.items():
            logger.info(f"  {name}: {counts['trainable']:,} trainable / {counts['total']:,} total")
            total_trainable += counts['trainable']
            total_params += counts['total']
        
        logger.info(f"Total: {total_trainable:,} trainable / {total_params:,} total ({total_trainable/total_params*100:.1f}%)")
        logger.info("=" * 50)
    
    #
    # ----------------- Prompt Encoding Logic (Copied and Adapted from pipeline_sd3.py) -----------------
    #
    
    @torch.no_grad()
    def _get_t5_prompt_embeds(
        self, prompt: List[str], num_images_per_prompt: int = 1
    ):
        """Helper to get T5 prompt embeds."""
        # Run on the text encoder's device to avoid moving encoders to GPU
        device = next(self.text_encoder_3.parameters()).device
        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        
        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]
        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds

    @torch.no_grad()
    def _get_clip_prompt_embeds(
        self, prompt: List[str], num_images_per_prompt: int = 1, clip_model_index: int = 0
    ):
        """Helper to get CLIP prompt embeds."""
        # Run on the text encoder's device to avoid moving encoders to GPU
        device = next(self.text_encoder.parameters()).device
        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2] # last-but-one layer

        # Move embeddings to main training device with the dtype used by transformer
        main_device = self.device
        main_dtype = self.models['transformer'].dtype
        prompt_embeds = prompt_embeds.to(dtype=main_dtype, device=main_device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=main_dtype, device=main_device)
        return prompt_embeds, pooled_prompt_embeds

    @torch.no_grad()
    def _encode_prompt(
        self,
        prompt: List[str],
        prompt_2: Optional[List[str]] = None,
        prompt_3: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
    ):
        """
        Encodes the prompt into text embeddings. This is a simplified version of the
        encode_prompt function from the original pipeline, adapted for training.
        """
        device = device or self.device

        # Handle optional prompts
        prompt_2 = prompt_2 or prompt
        prompt_3 = prompt_3 or prompt

        # Get CLIP embeddings
        prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
            prompt=prompt, num_images_per_prompt=num_images_per_prompt, clip_model_index=0
        )
        prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
            prompt=prompt_2, num_images_per_prompt=num_images_per_prompt, clip_model_index=1
        )
        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

        # Get T5 embeddings
        t5_prompt_embed = self._get_t5_prompt_embeds(
            prompt=prompt_3, num_images_per_prompt=num_images_per_prompt,
        )
        
        # Pad CLIP embeds to match T5 embeds
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )

        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed.to(device=self.device, dtype=clip_prompt_embeds.dtype)], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        return prompt_embeds, pooled_prompt_embeds

    def train(self):
        """The main training loop."""
        print("Starting Stage 1 training...")
        
        for epoch in range(self.config['stage1']['num_train_epochs']):
            print(f"--- Epoch {epoch+1}/{self.config['stage1']['num_train_epochs']} ---")
            
            for step, batch in enumerate(self.train_dataloader):
                with torch.cuda.amp.autocast(enabled=self.config['stage1']['use_amp']):
                    logger.info(f"Caculate loss...")
                    loss = self._train_step(batch)
                    logger.info(f"Loss: {loss}")

                    '''
                    # 跳过返回None的batch
                    if loss is None:
                        logger.warning(f"Skipping batch {step} due to processing error")
                        continue
                    '''
                    loss = loss / self.config['stage1']['gradient_accumulation_steps']

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (step + 1) % self.config['stage1']['gradient_accumulation_steps'] == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                self.global_step += 1
                
                if self.global_step % 10 == 0:  # Log progress
                    print(f"Step {self.global_step}, Loss: {loss.item() * self.config['stage1']['gradient_accumulation_steps']:.4f}")

                # 保存训练图像 (注释掉)
                if self.global_step % 2 == 0: os.makedirs('./training_output/training_images', exist_ok=True); save_image((batch['pixel_values'] / 2 + 0.5).clamp(0, 1), f'./training_output/training_images/step_{self.global_step}.png')

                # Validation and Checkpointing
                if self.global_step % self.config['stage1']['validation_steps'] == 0:
                    val_loss = self._validate()
                    
                    # Generate validation images for visual inspection
                    self._log_validation_images(epoch, self.global_step)
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        print("New best validation loss. Saving best model...")
                        self._save_checkpoint(epoch, is_best=True)
        
        print("Training finished.")
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            float: Average training loss for the epoch
        """
        # Set trainable models to train mode
        self.models['ids_text_embedder'].train()
        self.models['adapter'].train()
        
        # Set frozen models to eval mode
        self.models['vae'].eval()
        self.models['transformer'].eval()
        self.models['text_render_net'].eval()
        self.models['scene_gen_net'].eval()
        
        epoch_losses = []
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            try:
                loss = self._train_step(batch)
                epoch_losses.append(loss.item())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'step': self.global_step
                })
                
                # Log training step
                if self.global_step % 100 == 0:
                    logger.info(f"Step {self.global_step}: loss={loss.item():.6f}")
                
                self.global_step += 1
                
            except Exception as e:
                logger.error(f"Error in training step {step}: {e}")
                continue
        
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def _train_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Perform a single training step with forward pass and loss calculation.
        This method only handles forward pass and loss computation.
        Gradient computation and optimization are handled in the main training loop.
        
        Args:
            batch: Training batch from dataloader
            
        Returns:
            torch.Tensor: Computed loss value
        """
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        # Ensure correct dtypes for tokenized inputs to avoid overflows
        for key in ["input_ids", "token_type_ids", "attention_mask"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(dtype=torch.long, device=self.device)
        if "attention_mask" in batch and isinstance(batch["attention_mask"], torch.Tensor):
            batch["attention_mask"] = batch["attention_mask"].clamp(min=0, max=1)
        if "text_pos" in batch and isinstance(batch["text_pos"], torch.Tensor):
            batch["text_pos"] = batch["text_pos"].to(dtype=torch.float32, device=self.device)
        
        # Also move prompt if it's a list of strings
        if "prompt" in batch and isinstance(batch["prompt"], list):
            
            # The prompt itself is not a tensor, it will be processed by the tokenizer
            pass

        # Clamp token ids to valid ranges to avoid int64 overflow or embedding OOB
        try:
            if "input_ids" in batch and isinstance(batch["input_ids"], torch.Tensor):
                vocab_size = getattr(self.models['ids_text_embedder'].tokenizer, 'vocab_size', None)
                if vocab_size is not None:
                    batch["input_ids"] = batch["input_ids"].clamp_(min=0, max=vocab_size - 1)
            if "token_type_ids" in batch and isinstance(batch["token_type_ids"], torch.Tensor):
                batch["token_type_ids"] = batch["token_type_ids"].clamp_(min=0, max=1)
        except Exception as _:
            # Safe guard: if clamping fails due to dtype or missing attributes, continue
            pass
        logger.info("In _train_step(), before _forward_pass...")
        # Forward pass through the complete pipeline
        loss = self._forward_pass(batch)
        
        return loss
    
    def _forward_pass(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Perform forward pass through the model pipeline.
        Implements the complete diffusion training process for Stage 1.
        
        Args:
            batch: Training batch containing images, text, and conditioning data
            
        Returns:
            torch.Tensor: Computed loss
        """
        # 1. Encode images to latent space
        with torch.no_grad():
            # Scale factor for SD3 VAE (typically 0.13025)
            scaling_factor = getattr(self.models['vae'].config, 'scaling_factor', 0.13025)
            logger.info(f"Scaling factor: {scaling_factor}")
            logger.info(f"Batch pixel values shape: {batch['pixel_values'].shape}")
            latents = self.models['vae'].encode(batch['pixel_values']).latent_dist.sample()
            logger.info(f"Latents shape: {latents.shape}")
            latents = latents * scaling_factor
            logger.info(f"Latents shape after scaling: {latents.shape}")
        
        # 2. Sample noise and timesteps for diffusion process
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, 
            (latents.shape[0],), 
            device=self.device
        ).long()
        
        # 3. Add noise to latents (forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        logger.info(f"Noisy latents shape: {noisy_latents.shape}")
        
        # 4. Get text conditioning from our new IDS-based modules for TextRenderNet
        try:
            # Use all_texts_info which contains ALL texts from each sample
            batch_texts = batch['all_texts_info']  # List[List[Dict]] format expected by embedder
            logger.info(f"Processing {len(batch_texts)} samples with multiple texts each")
            
            # DEBUG: Print actual text content to verify we're loading all texts
            for i, sample_texts in enumerate(batch_texts):
                logger.info(f"Sample {i} has {len(sample_texts)} texts:")
                for j, text_info in enumerate(sample_texts):
                    logger.info(f"  Text {j}: '{text_info['content']}' at pos {text_info['pos']}")
                # Also check the combined content
                logger.info(f"Sample {i} combined content: '{batch['text_content'][i]}'")
            
            text_embeds = self.models['ids_text_embedder'].get_text_embeds_batch(
                batch_texts=batch_texts
            )  # (batch_size, 112, 128) - token-level features
            logger.info(f"Text embeds shape: {text_embeds.shape}")
            
            # Transform through adapter to get conditioning features for TextRenderNet
            text_render_features = self.models['adapter'](text_embeds)  # (batch_size, 112, projection_dim)
            logger.info(f"Text render features shape: {text_render_features.shape}")

        except Exception as e:
            logger.error(f"Error in text processing: {e}")
            logger.error(f"Batch all_texts_info format: {type(batch.get('all_texts_info', 'missing'))}")
            # Create fallback zero features to continue training
            batch_size = batch['pixel_values'].shape[0]
            projection_dim = self.models['transformer'].config.joint_attention_dim  # 4096
            text_render_features = torch.zeros(
                batch_size, 112, projection_dim,
                device=self.device, dtype=torch.float32
            )
            logger.warning("Using zero text features as fallback")
        
        # 4.bis. Get prompt conditioning for SceneGenNet and the main Transformer
        try:
            # The dataset provides a single 'prompt' list. We use it for all three encoders,
            # which is the standard behavior when prompt_2 and prompt_3 are not provided.
            prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt=batch['prompt'])
        except Exception as e:
            logger.error(f"Error in prompt encoding: {e}")
            return None

        # 5. Get ControlNet guidance from both ControlNets
        try:
            # Prepare pooled projections (zeros) to satisfy SD3 embedding API
            # Note: In training, we often use zeros for pooled projections for simplicity
            pooled_projection_dim = self.models['transformer'].config.pooled_projection_dim
            pooled_zeros = torch.zeros(
                noisy_latents.shape[0], pooled_projection_dim,
                device=noisy_latents.device, dtype=noisy_latents.dtype
            )

            # 5.1 Prepare ControlNet conditioning in latent space
            with torch.no_grad():
                # Encode scene conditioning image to latent space
                scene_cond_latents = self.models['vae'].encode(batch['scene_conditioning_pixel_values']).latent_dist.sample()
                scene_cond_latents = scene_cond_latents * scaling_factor
                
                # For SceneGenNet, we need to add mask channel (using conditioning_mask)
                # Resize mask to latent space dimensions
                conditioning_mask_resized = torch.nn.functional.interpolate(
                    batch['conditioning_mask'], 
                    size=(scene_cond_latents.shape[2], scene_cond_latents.shape[3]),
                    mode='nearest'
                )
                # Concatenate image latents (16 channels) + mask (1 channel) = 17 channels
                scene_controlnet_input = torch.cat([scene_cond_latents, conditioning_mask_resized], dim=1)
                
                # Encode text conditioning image to latent space (16 channels only)
                text_cond_latents = self.models['vae'].encode(batch['text_render_conditioning_pixel_values']).latent_dist.sample()
                text_cond_latents = text_cond_latents * scaling_factor
                text_controlnet_input = text_cond_latents

            # Get SceneGenNet residuals (expects 17 channels: 16 latent + 1 mask)
            scene_control_output = self.models['scene_gen_net'](
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_zeros,
                controlnet_cond=scene_controlnet_input,
                return_dict=False
            )
            scene_block_samples = scene_control_output[0]

            # Get TextRenderNet residuals (expects 16 channels: latent only)
            text_control_output = self.models['text_render_net'](
                hidden_states=noisy_latents,  
                timestep=timesteps,
                encoder_hidden_states=text_render_features,
                pooled_projections=pooled_zeros,
                controlnet_cond=text_controlnet_input,
                return_dict=False
            )
            text_block_samples = text_control_output[0]

            # Combine residuals from both controlnets
            # As per PosterMaker pipeline, we add the residuals together
            control_block_samples = []
            for scene_sample, text_sample in zip(scene_block_samples, text_block_samples):
                control_block_samples.append(scene_sample + text_sample)
            control_block_samples = tuple(control_block_samples)
                
        except Exception as e:
            logger.error(f"Error in ControlNet processing: {e}")
            # Fallback to no control guidance
            control_block_samples = None
        
        # 6. Predict noise with the main SD3 Transformer
        try:
            # CRITICAL FIX: Use our trainable text features instead of frozen prompt embeds
            # This ensures gradients flow through IDSTextEmbedder and Adapter
            model_pred = self.models['transformer'](
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=text_render_features,  # Use trainable features instead of frozen prompt_embeds!
                pooled_projections=pooled_prompt_embeds,
                block_controlnet_hidden_states=control_block_samples,  # Inject COMBINED ControlNet residuals
                return_dict=False
            )[0]
            
        except Exception as e:
            logger.error(f"Error in transformer prediction: {e}")
            raise
        
        # 7. Calculate loss
        loss = self._calculate_loss(model_pred, noise, batch)
        
        return loss
    
    def _calculate_loss(self, 
                       predicted_noise: torch.Tensor, 
                       target_noise: torch.Tensor, 
                       batch: Dict[str, Any]) -> torch.Tensor:
        """
        Calculate the loss for diffusion training.
        The base loss is the L2 loss (MSE) between the predicted and target noise.
        
        Args:
            predicted_noise: Noise predicted by the model
            target_noise: Ground truth noise
            batch: Training batch for additional loss terms
            
        Returns:
            torch.Tensor: Computed loss value
        """
        # Primary denoising loss (MSE between predicted and target noise)
        denoise_loss = F.mse_loss(
            predicted_noise.float(), 
            target_noise.float(), 
            reduction="mean"
        )
        
        # Extensibility point for future reward terms
        # Additional loss terms can be added here in future stages:
        # - Perceptual loss for better visual quality
        # - Adversarial loss for improved realism  
        # - Text-image alignment loss
        # - Region-specific loss weighting based on masks
        
        total_loss = denoise_loss
        
        return total_loss
    
    @torch.no_grad()  # Disable gradient calculation for validation
    def _validate(self) -> float:
        """Performs a full validation loop."""
        self.models['ids_text_embedder'].eval()
        self.models['adapter'].eval()
        
        total_val_loss = 0
        for batch in self.val_dataloader:
            # The validation step logic is identical to the training step,
            # but without backpropagation. We reuse _train_step for simplicity.
            loss = self._train_step(batch)
            total_val_loss += loss.item()
            
        avg_val_loss = total_val_loss / len(self.val_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Switch back to train mode
        self.models['ids_text_embedder'].train()
        self.models['adapter'].train()
        
        return avg_val_loss
    
    @torch.no_grad()
    def _log_validation_images(self, epoch: int, step: int, num_images: int = 4):
        """
        Generates and saves validation images to visually inspect model performance.
        Performs a complete diffusion inference denoising loop.

        Args:
            epoch (int): The current epoch, for naming the output file.
            step (int): The current global step, for naming the output file.
            num_images (int): The number of images to generate.
        """
        print(f"Generating validation images at step {step}...")
        
        # Set all models to evaluation mode
        self.models['vae'].eval()
        self.models['transformer'].eval()
        self.models['text_render_net'].eval()
        self.models['scene_gen_net'].eval()
        self.models['ids_text_embedder'].eval()
        self.models['adapter'].eval()

        try:
            # Get a fixed batch of validation data for consistent comparison
            val_batch = next(iter(self.val_dataloader))

            # Move batch to device
            for key, value in val_batch.items():
                if isinstance(value, torch.Tensor):
                    val_batch[key] = value.to(self.device)
            
            # Limit to num_images for efficiency
            batch_size = min(val_batch['pixel_values'].shape[0], num_images)
            for key, value in val_batch.items():
                if isinstance(value, torch.Tensor):
                    val_batch[key] = value[:batch_size]
                elif isinstance(value, list):
                    val_batch[key] = value[:batch_size]

            # --- Prepare Text Conditioning ---
            if 'text_pos' in val_batch:
                if isinstance(val_batch['text_pos'], torch.Tensor):
                    val_batch['text_pos'] = val_batch['text_pos'].to(device=self.device, dtype=torch.float32)
                else:
                    val_batch['text_pos'] = torch.tensor(val_batch['text_pos'], device=self.device, dtype=torch.float32)
            
            # Use all_texts_info for validation (contains ALL texts from each sample)
            val_batch_texts = val_batch['all_texts_info']
            
            text_embeds = self.models['ids_text_embedder'].get_text_embeds_batch(
                batch_texts=val_batch_texts
            )
            
            # Transform through adapter to get conditioning features
            text_features = self.models['adapter'](text_embeds)  # (batch_size, 112, projection_dim)

            # --- Prepare Prompt Conditioning ---
            # Use the same prompt for all encoders, mirroring the training setup
            prompt_embeds, pooled_prompt_embeds = self._encode_prompt(val_batch['prompt'])
            
            # Prepare ControlNet conditioning images
            scene_cond = val_batch['scene_conditioning_pixel_values']
            text_cond = val_batch['text_render_conditioning_pixel_values']

            # --- Initialize Random Noise Latents ---
            # Get the correct latent shape by encoding a sample image
            with torch.no_grad():
                sample_latents = self.models['vae'].encode(val_batch['pixel_values']).latent_dist.sample()
                scaling_factor = getattr(self.models['vae'].config, 'scaling_factor', 0.13025)
                sample_latents = sample_latents * scaling_factor
            
            # Initialize random noise with the same shape
            latents = torch.randn_like(sample_latents)

            # --- Set up Diffusion Scheduler for Inference ---
            # Set timesteps for the scheduler (fewer steps for faster validation)
            num_inference_steps = 20  # Reduced for faster validation
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.scheduler.timesteps

            # --- Full Denoising Loop ---
            print(f"Running denoising loop with {num_inference_steps} steps...")
            for i, t in enumerate(auto_tqdm(timesteps, desc="Denoising", leave=False)):
                # Expand timestep to match batch size
                timestep = t.expand(latents.shape[0])
                
                try:
                    # Prepare pooled projections (zeros)
                    pooled_projection_dim = self.models['transformer'].config.pooled_projection_dim
                    pooled_zeros = torch.zeros(
                        latents.shape[0], pooled_projection_dim,
                        device=latents.device, dtype=latents.dtype
                    )

                    # Get SceneGenNet guidance
                    scene_control_output = self.models['scene_gen_net'](
                        hidden_states=latents,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_zeros,
                        controlnet_cond=scene_cond,
                        return_dict=False
                    )
                    scene_block_samples = scene_control_output[0]

                    # Get TextRenderNet guidance
                    text_control_output = self.models['text_render_net'](
                        hidden_states=latents,
                        timestep=timestep,
                        encoder_hidden_states=text_features,
                        pooled_projections=pooled_zeros,
                        controlnet_cond=text_cond,
                        return_dict=False
                    )
                    text_block_samples = text_control_output[0]
                    
                    # Combine residuals
                    control_block_samples = []
                    for scene_sample, text_sample in zip(scene_block_samples, text_block_samples):
                        control_block_samples.append(scene_sample + text_sample)
                    control_block_samples = tuple(control_block_samples)

                    # Predict noise with main transformer
                    noise_pred = self.models['transformer'](
                        hidden_states=latents,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        block_controlnet_hidden_states=control_block_samples,  # Inject combined residuals
                        return_dict=False
                    )[0]
                    
                    # Update latents using scheduler
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    
                except Exception as e:
                    print(f"Error in denoising step {i}: {e}")
                    # Continue with next timestep or break if critical
                    continue

            # --- Decode Latents to Images ---
            print("Decoding latents to images...")
            
            # Scale latents back to original range
            latents = latents / scaling_factor
            
            # Decode to pixel space
            with torch.no_grad():
                images = self.models['vae'].decode(latents).sample
            
            # Post-process images: denormalize from [-1, 1] to [0, 1]
            images = (images / 2 + 0.5).clamp(0, 1)
            
            # --- Save Images ---
            output_dir = os.path.join(self.config['output_dir'], 'validation_samples')
            os.makedirs(output_dir, exist_ok=True)
            
            # Save generated images
            save_path = os.path.join(output_dir, f"generated_epoch_{epoch}_step_{step}.png")
            save_image(images, save_path, nrow=2, normalize=False)
            
            # Also save the conditioning images for comparison
            scene_cond_save_path = os.path.join(output_dir, f"scene_cond_epoch_{epoch}_step_{step}.png")
            scene_cond_images = (scene_cond / 2 + 0.5).clamp(0, 1)  # Denormalize
            save_image(scene_cond_images, scene_cond_save_path, nrow=2, normalize=False)

            text_cond_save_path = os.path.join(output_dir, f"text_cond_epoch_{epoch}_step_{step}.png")
            text_cond_images = (text_cond / 2 + 0.5).clamp(0, 1)  # Denormalize
            save_image(text_cond_images, text_cond_save_path, nrow=2, normalize=False)
            
            # Save original images for reference
            original_save_path = os.path.join(output_dir, f"original_epoch_{epoch}_step_{step}.png")
            original_images = (val_batch['pixel_values'] / 2 + 0.5).clamp(0, 1)  # Denormalize
            save_image(original_images, original_save_path, nrow=2, normalize=False)
            
            print(f"✓ Validation images saved:")
            print(f"  Generated: {save_path}")
            print(f"  Scene Conditioning: {scene_cond_save_path}")
            print(f"  Text Conditioning: {text_cond_save_path}")
            print(f"  Original: {original_save_path}")

        except Exception as e:
            print(f"Error in validation image generation: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Always switch trainable models back to training mode
            self.models['ids_text_embedder'].train()
            self.models['adapter'].train()
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Saves the trainable model weights."""
        output_dir = os.path.join(self.config['output_dir'], 'checkpoints')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save only the state dicts of the models we are training
        trainable_models = {
            "ids_text_embedder": self.models['ids_text_embedder'],
            "adapter": self.models['adapter']
        }
        
        if is_best:
            save_path = os.path.join(output_dir, "best_model.pth")
        else:
            save_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
            
        # Create a dictionary to save
        checkpoint = {name: model.state_dict() for name, model in trainable_models.items()}
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def _save_training_metrics(self):
        """Save training metrics to JSON file."""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss
        }
        
        metrics_path = self.log_dir / 'training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        self.models['ids_text_embedder'].load_state_dict(
            checkpoint['model_states']['ids_text_embedder']
        )
        self.models['adapter'].load_state_dict(
            checkpoint['model_states']['adapter']
        )
        
        # Load optimizer state
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Load scheduler state
        if 'scheduler_state' in checkpoint and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        # Restore training state
        self.global_step = checkpoint.get('step', 0)
        self.current_epoch = checkpoint.get('epoch', 0)
        
        logger.info(f"Checkpoint loaded. Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get current training statistics.
        
        Returns:
            Dict containing training statistics
        """
        return {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'total_params': sum(count_trainable_parameters(self.models)[name]['total'] 
                              for name in self.models),
            'trainable_params': sum(count_trainable_parameters(self.models)[name]['trainable'] 
                                  for name in self.models)
        }