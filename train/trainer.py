"""
Trainer
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
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
        self.output_dir = Path(config.get('output_dir', '../training_output'))
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        
        # Create output directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        self._setup_logging()
        self._log_initialization_info()
        
        # logger.info("Trainer initialized successfully for training")
    
    def _setup_logging(self):
        log_file = self.log_dir / f'training_{int(time.time())}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.info(f"Training logs will be saved to: {log_file}")
    
    def _count_parameters(self):
        param_counts = {}
        for name, model in self.models.items():
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            param_counts[name] = {'trainable': trainable, 'total': total}
        return param_counts
    
    def _log_initialization_info(self):
        """Log important information about the training setup."""
        # Log model parameter counts
        param_counts = self._count_parameters()
        
        logger.info("==== Training Setup Information ====")
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
        
        # logger.info(f"Total: {total_trainable:,} trainable / {total_params:,} total ({total_trainable/total_params*100:.1f}%)")
        logger.info("=" * 50)
    
    #
    # Prompt Encoding
    #
    
    @torch.no_grad()
    def _get_t5_prompt_embeds(
        self, prompt: List[str], num_images_per_prompt: int = 1
    ):
        """Helper to get T5 prompt embeds."""
        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        
        # Text encoder is on CPU, process there then move result to GPU
        prompt_embeds = self.text_encoder_3(text_input_ids.to("cpu"))[0]
        # Keep embeddings in FP32 for compatibility with trainable models
        prompt_embeds = prompt_embeds.to(dtype=torch.float32, device=self.device)

        return prompt_embeds

    @torch.no_grad()
    def _get_clip_prompt_embeds(
        self, prompt: List[str], num_images_per_prompt: int = 1, clip_model_index: int = 0
    ):
        """Helper to get CLIP prompt embeds."""
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
        
        # Text encoder is on CPU, process there then move result to GPU
        prompt_embeds = text_encoder(text_input_ids.to("cpu"), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]

        # Move results to GPU - keep in FP32 for compatibility with trainable models
        prompt_embeds = prompt_embeds.to(dtype=torch.float32, device=self.device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=torch.float32, device=self.device)
        
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

        device = device or self.device

        prompt_2 = prompt_2 or prompt
        prompt_3 = prompt_3 or prompt

        prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
            prompt=prompt, num_images_per_prompt=num_images_per_prompt, clip_model_index=0
        )
        prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
            prompt=prompt_2, num_images_per_prompt=num_images_per_prompt, clip_model_index=1
        )
        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

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

        logger.info("Starting training...")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            logger.info(f"Starting epoch {epoch+1}/{self.num_epochs}")
            print(f"--- Epoch {epoch+1}/{self.num_epochs} ---")
            
            avg_train_loss = self._train_epoch()
            
            if not math.isnan(avg_train_loss):
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
        self.models['text_render_net'].train()  # Partially trainable
        
        # Set frozen models to eval mode and enable gradient checkpointing for memory efficiency
        self.models['vae'].eval()
        self.models['transformer'].eval()
        self.models['scene_gen_net'].eval()
        
        # Enable gradient checkpointing on trainable models to save memory
        if hasattr(self.models['text_render_net'], 'enable_gradient_checkpointing'):
            self.models['text_render_net'].enable_gradient_checkpointing()
        if hasattr(self.models['ids_text_embedder'], 'enable_gradient_checkpointing'):
            self.models['ids_text_embedder'].enable_gradient_checkpointing()
        if hasattr(self.models['adapter'], 'enable_gradient_checkpointing'):
            self.models['adapter'].enable_gradient_checkpointing()
        
        # Enable gradient checkpointing on frozen models for memory efficiency
        if hasattr(self.models['transformer'], 'enable_gradient_checkpointing'):
            self.models['transformer'].enable_gradient_checkpointing()
        if hasattr(self.models['scene_gen_net'], 'enable_gradient_checkpointing'):
            self.models['scene_gen_net'].enable_gradient_checkpointing()
        
        epoch_losses = []
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            try:
                # Use autocast when AMP is enabled
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        loss = self._train_step(batch)
                        loss = loss / self.gradient_accumulation_steps
                else:
                    loss = self._train_step(batch)
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with scaler when AMP enabled
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Check training health BEFORE clearing gradients
                    accumulated_loss = loss.item() * self.gradient_accumulation_steps
                    is_healthy, grad_norm, warnings = self._check_training_health(accumulated_loss)
                    if not is_healthy:
                        if self.global_step % 10 == 0:  # Don't spam warnings
                            print(f"⚠️  Training Health Warning at step {self.global_step}:")
                            for warning in warnings:
                                print(f"    - {warning}")
                        
                        # Skip training step if gradients are exploding
                        if grad_norm > 10.0 or not torch.isfinite(torch.tensor(grad_norm)):
                            logger.warning(f"Skipping step due to gradient explosion: {grad_norm}")
                            self.optimizer.zero_grad()
                            torch.cuda.empty_cache()
                            continue
                    
                    # Optimizer step with or without scaler
                    if self.use_amp:
                        if self.stage1_config.get('max_grad_norm', None):
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                [p for name in ['ids_text_embedder', 'adapter', 'text_render_net'] for p in self.models[name].parameters() if p.requires_grad],
                                self.stage1_config['max_grad_norm']
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if self.stage1_config.get('max_grad_norm', None):
                            torch.nn.utils.clip_grad_norm_(
                                [p for name in ['ids_text_embedder', 'adapter', 'text_render_net'] for p in self.models[name].parameters() if p.requires_grad],
                                self.stage1_config['max_grad_norm']
                            )
                        self.optimizer.step()
                    
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
                
                # Aggressive memory cleanup every step
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg:
                    logger.error(f"CUDA OOM at step {step}. Skipping batch and clearing cache.")
                    # Aggressive cleanup on OOM
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()  # Second pass
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
        
        # Ensure we never return None to avoid cascade errors
        if loss is None:
            logger.error("Forward pass returned None, using dummy loss")
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
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
            # SD3 VAE encoding: (latents - shift_factor) * scaling_factor
            scaling_factor = getattr(self.models['vae'].config, 'scaling_factor', 1.5305)
            shift_factor = getattr(self.models['vae'].config, 'shift_factor', 0.0609)
            
            # VAE is on GPU with FP16, convert data accordingly
            pixel_values = batch['pixel_values'].to(device=self.device, dtype=torch.float16)
            latents = self.models['vae'].encode(pixel_values).latent_dist.sample()
            latents = (latents - shift_factor) * scaling_factor
            
            # Clear pixel_values immediately to save memory
            del pixel_values
        
        # 2. Sample noise and timesteps for diffusion process
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, 
            (latents.shape[0],), 
            device=self.device
        ).long()
        
        # 3. Add noise to latents (forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Clear intermediate variables to save memory
        torch.cuda.empty_cache()
        
        # 4. Get text conditioning from IDS-based modules for TextRenderNet
        batch_size = batch['pixel_values'].shape[0]
        projection_dim = self.models['transformer'].config.joint_attention_dim
        
        try:
            # IDS text embedder expects List[List[Dict]], but we have List[Dict] per sample
            # Need to wrap each sample's texts in a list for the batch
            batch_texts = [batch['all_texts_info']]  # Wrap single sample's texts in a list
            text_embeds = self.models['ids_text_embedder'].get_text_embeds_batch(
                batch_texts=batch_texts
            )
            # Ensure dtype consistency for text processing pipeline
            texts_embeds = text_embeds.to(device=self.device, dtype=torch.float32)  # Keep FP32 for processing
            texts_embeds = self.models['adapter'](texts_embeds)
            # text_render_net is now FP32, so no conversion needed
            texts_embeds = texts_embeds.to(dtype=torch.float32)

        except Exception as e:
            logger.error(f"Error in text processing: {e}")
            texts_embeds = torch.zeros(
                batch_size, 112, projection_dim,
                device=self.device, dtype=torch.float32
            )
        
        # 4.bis. Get prompt conditioning for SceneGenNet and the main Transformer
        try:
            prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt=batch['prompt'])
        except Exception as e:
            logger.error(f"Error in prompt encoding: {e}")
            # Return fallback embeddings instead of None to avoid cascade errors
            seq_len = 154  # Standard SD3 sequence length (77 + 77)
            hidden_dim = self.models['transformer'].config.joint_attention_dim
            pooled_dim = self.models['transformer'].config.pooled_projection_dim
            
            prompt_embeds = torch.zeros(
                batch_size, seq_len, hidden_dim,
                device=self.device, dtype=torch.float16
            )
            pooled_prompt_embeds = torch.zeros(
                batch_size, pooled_dim,
                device=self.device, dtype=torch.float16
            )

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
                scene_cond_pixels = batch['scene_conditioning_pixel_values'].to(device=self.device, dtype=torch.float16)
                scene_cond_latents = self.models['vae'].encode(scene_cond_pixels).latent_dist.sample()
                scene_cond_latents = (scene_cond_latents - shift_factor) * scaling_factor
                del scene_cond_pixels  # Free memory immediately
                
                # Resize subject mask to latent space dimensions
                subject_mask_resized = torch.nn.functional.interpolate(
                    batch['subject_mask'], 
                    size=(scene_cond_latents.shape[2], scene_cond_latents.shape[3]),
                    mode='nearest'
                )
                scene_controlnet_input = torch.cat([scene_cond_latents, subject_mask_resized], dim=1)
                
                # Encode text conditioning image to latent space
                text_cond_pixels = batch['text_render_conditioning_pixel_values'].to(device=self.device, dtype=torch.float16)
                text_cond_latents = self.models['vae'].encode(text_cond_pixels).latent_dist.sample()
                text_cond_latents = (text_cond_latents - shift_factor) * scaling_factor
                text_controlnet_input = text_cond_latents
                del text_cond_pixels  # Free memory immediately

            # Get SceneGenNet residuals
            # scene_gen_net is FP16
            scene_noisy_latents = noisy_latents.to(dtype=torch.float16)
            scene_timesteps = timesteps.to(device=scene_noisy_latents.device)
            scene_prompt_embeds = prompt_embeds.to(dtype=torch.float16)
            scene_pooled_zeros = pooled_zeros.to(dtype=torch.float16)
            scene_controlnet_input = scene_controlnet_input.to(dtype=torch.float16)
            
            scene_control_output = self.models['scene_gen_net'](
                hidden_states=scene_noisy_latents,
                timestep=scene_timesteps,
                encoder_hidden_states=scene_prompt_embeds,
                pooled_projections=scene_pooled_zeros,
                controlnet_cond=scene_controlnet_input,
                conditioning_scale=1.0,
                return_dict=False
            )
            scene_block_samples = scene_control_output[0]

            # Get TextRenderNet residuals  
            # text_render_net is now FP32
            text_noisy_latents = noisy_latents.to(dtype=torch.float32)
            text_timesteps = timesteps.to(device=text_noisy_latents.device)
            text_pooled_zeros = pooled_zeros.to(dtype=torch.float32)
            text_controlnet_input = text_controlnet_input.to(dtype=torch.float32)
            
            text_control_output = self.models['text_render_net'](
                hidden_states=text_noisy_latents,  
                timestep=text_timesteps,
                encoder_hidden_states=texts_embeds,  # Already converted to proper dtype above
                pooled_projections=text_pooled_zeros,
                controlnet_cond=text_controlnet_input,
                conditioning_scale=1.0,
                return_dict=False
            )
            text_block_samples = text_control_output[0]

            
            # Ensure compatible dtypes before combining block samples
            # Convert text_block_samples to FP16 to match scene_block_samples
            text_block_samples = [sample.to(dtype=torch.float16) for sample in text_block_samples]
            
            block_interval = (len(scene_block_samples) + 1) // len(text_block_samples)
            control_block_samples = []
            for block_i in range(len(scene_block_samples)):
                control_block_sample = scene_block_samples[block_i] + text_block_samples[block_i // block_interval]
                control_block_samples.append(control_block_sample)
            
            # Clear intermediate ControlNet variables
            del scene_block_samples, text_block_samples
            torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error in ControlNet processing: {e}")
            # Fallback to no control guidance with proper dtype
            control_block_samples = None
        
        # 6. Predict noise with the main SD3 Transformer
        try:
            # Main SD3 uses standard CLIP+T5 prompt encoding
            # IDS text features influence through TextRenderNet residuals, not direct input
            # Ensure all inputs match transformer dtype (FP16)
            noisy_latents = noisy_latents.to(dtype=torch.float16)
            timesteps = timesteps.to(device=noisy_latents.device)
            prompt_embeds = prompt_embeds.to(dtype=torch.float16)
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=torch.float16)
            
            # Ensure ControlNet outputs match transformer dtype if they exist
            if control_block_samples is not None:
                if isinstance(control_block_samples, (list, tuple)):
                    control_block_samples = [s.to(dtype=torch.float16) for s in control_block_samples]
                else:
                    control_block_samples = control_block_samples.to(dtype=torch.float16)
            
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
        # Check gradients for all trainable models
        for model_name in ['ids_text_embedder', 'adapter', 'text_render_net']:
            for name, param in self.models[model_name].named_parameters():
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
        if 'text_mask' in batch:
            text_mask = batch['text_mask'].float()  # Ensure float type
            
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
                masked_predicted, 
                masked_target, 
                reduction="sum"
            ) / (text_mask_latent.sum() + 1e-8)  # Avoid division by zero
            
            
        else:
            # Fallback: calculate loss on full image if no mask available
            total_loss = F.mse_loss(
                predicted_noise, 
                target_noise, 
                reduction="mean"
            )
        
        
        return total_loss
    
    @torch.no_grad()  # Disable gradient calculation for validation
    def _validate(self) -> float:
        """Performs a full validation loop."""
        self.models['ids_text_embedder'].eval()
        self.models['adapter'].eval()
        self.models['text_render_net'].eval()  # Also set to eval for validation
        self.models['vae'].eval()
        self.models['transformer'].eval()
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
        self.models['text_render_net'].train()  # Back to train mode
        self.models['vae'].eval()
        self.models['transformer'].eval()
        self.models['scene_gen_net'].eval()
        
        return avg_val_loss
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Saves the complete training state including model weights and training progress."""
        output_dir = os.path.join(self.config['output_dir'], 'checkpoints')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save trainable model state dicts
        trainable_models = {
            "ids_text_embedder": self.models['ids_text_embedder'],
            "adapter": self.models['adapter'],
            "text_render_net": self.models['text_render_net']
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

def main():
    """Main training function"""
    from setup import setup_all
    
    config_path = 'config.yaml'
    
    try:
        print("trainer start.")
        # Setup all components
        components = setup_all(config_path)
        
        # Initialize trainer
        trainer = Trainer(
            config=components['config'],
            models=components['models'],
            tokenizers=components['tokenizers'],
            train_dataloader=components['train_dataloader'],
            val_dataloader=components['val_dataloader'],
            optimizer=components['optimizer'],
            lr_scheduler=components['lr_scheduler'],
            device=components['device'],
            scheduler=components['scheduler']
        )
        
        trainer.train()

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()