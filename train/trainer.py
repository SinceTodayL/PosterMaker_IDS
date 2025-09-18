"""
Trainer
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
import numpy as np
import math

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
        
        self.stage1_config = config['stage1']
        self.num_epochs = self.stage1_config['num_train_epochs']
        self.gradient_accumulation_steps = self.stage1_config['gradient_accumulation_steps']
        self.save_steps = self.stage1_config['save_steps']
        self.validation_steps = self.stage1_config['validation_steps']
        self.use_amp = self.stage1_config['use_amp']
        self.validation_mode = self.stage1_config.get('validation_mode', 'step')
        
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
        
        logger.info("Trainer initialized successfully for training")
    
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
        logger.info(f"Validation Mode: {self.validation_mode}")
        logger.info(f"Save Steps: {self.save_steps}")
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
    # Prompt Encoding Logic (Copied and Adapted from pipeline_sd3.py) 
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
        logger.info("Starting training...")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            logger.info(f"Starting epoch {epoch+1}/{self.num_epochs}")
            print(f"--- Epoch {epoch+1}/{self.num_epochs} ---")
            
            avg_train_loss = self._train_epoch()
            
            if not math.isnan(avg_train_loss):  # Check if training was successful
                logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_train_loss:.6f}")
                
                if self.validation_mode == 'epoch':
                    val_loss = self._validate()
                    self.val_losses.append(val_loss)
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        logger.info(f"New best validation loss: {val_loss:.6f}. Saving best model...")
                        print("New best validation loss. Saving best model...")
                        self._save_checkpoint(epoch, is_best=True)
            else:
                logger.error(f"Epoch {epoch+1} failed completely (NaN loss). Skipping validation.")
                print(f"❌ Epoch {epoch+1} failed - no successful training steps!")
            
            # Note: save_steps should be handled per training step, not per epoch
            # This will be handled in the training step loop instead
            
            self._save_training_metrics()
        
        # Save final checkpoint
        logger.info("Saving final checkpoint...")
        self._save_checkpoint(self.current_epoch, is_best=False)
        
        logger.info("Training finished.")
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
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss = self._train_step(batch)
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Check training health BEFORE clearing gradients
                    if self.global_step % 100 == 0:
                        accumulated_loss = loss.item() * self.gradient_accumulation_steps
                        is_healthy, grad_norm, warnings = self._check_training_health(accumulated_loss)
                        if not is_healthy:
                            print(f"⚠️  Training Health Warning at step {self.global_step}:")
                            for warning in warnings:
                                print(f"    - {warning}")
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Record loss only after gradient accumulation is complete
                    accumulated_loss = loss.item() * self.gradient_accumulation_steps
                    epoch_losses.append(accumulated_loss)
                    
                    # Record learning rate for tracking
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.learning_rates.append(current_lr)
                    
                    if self.global_step % 10 == 0:
                        logger.info(f"Step {self.global_step}: loss={accumulated_loss:.6f}")
                        print(f"Step {self.global_step}, Loss: {accumulated_loss:.4f}")
                    
                    if self.validation_mode == 'step' and self.global_step % self.validation_steps == 0:
                        val_loss = self._validate()
                        self.val_losses.append(val_loss)
                        
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            logger.info(f"New best validation loss: {val_loss:.6f}. Saving best model...")
                            print("New best validation loss. Saving best model...")
                            self._save_checkpoint(self.current_epoch, is_best=True)
                    
                    # Save checkpoint based on training steps (not epochs)
                    if self.global_step % self.save_steps == 0:
                        logger.info(f"Saving checkpoint at step {self.global_step}")
                        self._save_checkpoint(self.current_epoch, is_best=False)
                
                # Update progress bar
                current_step = self.global_step + (step + 1) // self.gradient_accumulation_steps
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.6f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'step': current_step,
                    'acc': f'{(step + 1) % self.gradient_accumulation_steps}/{self.gradient_accumulation_steps}'
                })
                
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg:
                    logger.error(f"CUDA OOM at step {step}. Skipping batch and clearing cache.")
                    torch.cuda.empty_cache()
                    # Reset gradient accumulation to maintain consistency
                    self.optimizer.zero_grad()
                    continue
                elif "nan" in error_msg or "inf" in error_msg:
                    logger.error(f"NaN/Inf detected at step {step}: {e}")
                    # Reset gradient accumulation and continue
                    self.optimizer.zero_grad()
                    continue
                else:
                    # For other RuntimeErrors, re-raise as they might be serious
                    logger.error(f"Serious runtime error at step {step}: {e}")
                    raise
            except Exception as e:
                # For non-runtime errors, log and re-raise
                logger.error(f"Unexpected error at step {step}: {e}")
                raise
        
        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            self.train_losses.append(avg_loss)
            return avg_loss
        else:
            # No successful training steps in this epoch
            logger.warning(f"Epoch {self.current_epoch + 1} had no successful training steps!")
            # Don't add to train_losses to avoid misleading metrics
            return float('nan')  # Return NaN to indicate failed epoch
    
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
        # Forward pass through the complete pipeline
        loss = self._forward_pass(batch)
        
        return loss
    
    def _forward_pass(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
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
            
            # DEBUG: Print actual text content to verify
            for i, sample_texts in enumerate(batch_texts):
                logger.info(f"Sample {i} has {len(sample_texts)} texts:")
                for j, text_info in enumerate(sample_texts):
                    logger.info(f"  Text {j}: '{text_info['content']}' at pos {text_info['pos']}")
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
            # In training, we use zeros for pooled projections
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
            # Main SD3 uses standard CLIP+T5 prompt encoding
            # IDS text features influence through TextRenderNet residuals, not direct input
            model_pred = self.models['transformer'](
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,  # Use standard CLIP+T5 prompt embeds
                pooled_projections=pooled_prompt_embeds,
                block_controlnet_hidden_states=control_block_samples,  # IDS features affect through ControlNet residuals
                return_dict=False
            )[0]
            
        except Exception as e:
            logger.error(f"Error in transformer prediction: {e}")
            raise
        # 8. Calculate loss
        loss = self._calculate_loss(model_pred, noise, batch)
        
        return loss
    
    def _check_training_health(self, current_loss: float):
        """
        Lightweight check to ensure model is healthily training.
        
        Args:
            current_loss: Current training loss for trend analysis
        """
        # Check gradient norms to ensure gradients are flowing
        total_grad_norm = 0
        grad_count = 0
        for name, param in self.models['ids_text_embedder'].named_parameters():
            if param.requires_grad and param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
                grad_count += 1
        
        for name, param in self.models['adapter'].named_parameters():
            if param.requires_grad and param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
                grad_count += 1
        
        if grad_count > 0:
            total_grad_norm = (total_grad_norm / grad_count) ** 0.5
        else:
            total_grad_norm = 0.0
        
        # Check if training appears healthy
        is_healthy = True
        warnings = []
        
        if total_grad_norm < 1e-7:
            is_healthy = False
            warnings.append("Gradient norm too small - model may not be training")
        elif total_grad_norm > 10.0:
            warnings.append("Gradient norm very large - potential gradient explosion")
        
        # Track step-level losses for trend analysis
        if not hasattr(self, '_step_losses'):
            self._step_losses = []
        self._step_losses.append(current_loss)
        
        # Keep only recent losses (last 50 steps)
        if len(self._step_losses) > 50:
            self._step_losses = self._step_losses[-50:]
        
        # Check loss trend (if we have enough step history)
        if len(self._step_losses) >= 20:
            recent_losses = self._step_losses[-20:]
            # Use linear regression to check trend
            x = list(range(len(recent_losses)))
            y = recent_losses
            n = len(recent_losses)
            slope = (n * sum(xi * yi for xi, yi in zip(x, y)) - sum(x) * sum(y)) / (n * sum(xi**2 for xi in x) - sum(x)**2)
            
            if slope > 0.01:  # Loss increasing
                warnings.append("Loss trend increasing over last 20 steps")
        
        # Check for loss spikes (current loss much higher than recent average)
        if len(self._step_losses) >= 10:
            recent_avg = sum(self._step_losses[-10:-1]) / 9  # Exclude current loss
            if current_loss > recent_avg * 2.0:
                warnings.append(f"Loss spike detected: {current_loss:.4f} vs recent avg {recent_avg:.4f}")
        
        # Log results
        logger.info(f"Training Health Check - Step {self.global_step}:")
        logger.info(f"  Gradient norm: {total_grad_norm:.6f}")
        logger.info(f"  Current loss: {current_loss:.6f}")
        if warnings:
            for warning in warnings:
                logger.warning(f"  ⚠️  {warning}")
        else:
            logger.info("  ✅ Training appears healthy")
        
        return is_healthy, total_grad_norm, warnings
    
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
        if 'conditioning_mask' in batch:
            logger.info("Calculating loss with conditioning mask.")
            text_mask = batch['conditioning_mask'].float()  # Ensure float type
            
            # Resize mask to match latent dimensions
            latent_h, latent_w = predicted_noise.shape[2], predicted_noise.shape[3]
            text_mask_latent = torch.nn.functional.interpolate(
                text_mask, 
                size=(latent_h, latent_w), 
                mode='nearest'
            )
            
            # Apply mask to both predicted and target noise (only calculate loss on text regions)
            masked_predicted = predicted_noise * text_mask_latent
            masked_target = target_noise * text_mask_latent
            
            # Calculate MSE loss only on masked regions
            # Use sum reduction and normalize by the number of masked pixels
            total_loss = F.mse_loss(
                masked_predicted.float(), 
                masked_target.float(), 
                reduction="sum"
            ) / (text_mask_latent.sum() + 1e-8)  # Avoid division by zero
            
            print(f"Text mask coverage: {text_mask_latent.mean().item():.4f} ({text_mask_latent.sum().item():.0f} pixels)")
            
        else:
            # Fallback: calculate loss on full image if no mask available
            total_loss = F.mse_loss(
                predicted_noise.float(), 
                target_noise.float(), 
                reduction="mean"
            )
            print("Warning: No conditioning_mask found, using full image loss")
        
        
        return total_loss
    
    @torch.no_grad()  # Disable gradient calculation for validation
    def _validate(self) -> float:
        """Performs a full validation loop."""
        self.models['ids_text_embedder'].eval()
        self.models['adapter'].eval()
        self.models['vae'].eval()
        self.models['transformer'].eval()
        self.models['text_render_net'].eval()
        self.models['scene_gen_net'].eval()
        
        total_val_loss = 0
        for batch in self.val_dataloader:
            # The validation step logic is identical to the training step,
            # but without backpropagation. We reuse _train_step for simplicity.
            loss = self._train_step(batch)
            total_val_loss += loss.item()
            
        avg_val_loss = total_val_loss / len(self.val_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Restore model modes: trainable models to train, frozen models to eval
        self.models['ids_text_embedder'].train()
        self.models['adapter'].train()
        self.models['vae'].eval()
        self.models['transformer'].eval()
        self.models['text_render_net'].eval()
        self.models['scene_gen_net'].eval()
        
        return avg_val_loss
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Saves the complete training state including model weights and training progress."""
        output_dir = os.path.join(self.config['output_dir'], 'checkpoints')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save trainable model state dicts
        trainable_models = {
            "ids_text_embedder": self.models['ids_text_embedder'],
            "adapter": self.models['adapter']
        }
        
        if is_best:
            save_path = os.path.join(output_dir, "best_model.pth")
        else:
            save_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
            
        # Create complete checkpoint including training state
        checkpoint = {
            # Model states
            'model_state_dicts': {name: model.state_dict() for name, model in trainable_models.items()},
            
            # Training states
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            
            # Training progress
            'epoch': epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            
            # Training metrics
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            
            # Random states for reproducibility
            'random_state': torch.get_rng_state(),
            'numpy_random_state': np.random.get_state() if 'numpy' in globals() else None,
            
            # Config for reference
            'config': self.config
        }
        
        torch.save(checkpoint, save_path)
        print(f"Complete checkpoint saved to {save_path}")
    
    
    
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