"""
Training engine for PosterMaker Stage 1 IDS-based fine-tuning.
This module provides the main Trainer class that handles the training loop,
validation, checkpointing, and progress monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import os
import logging
import time
from pathlib import Path
from tqdm import tqdm
from tqdm.auto import tqdm as auto_tqdm
import json
from torchvision.utils import save_image

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
                 tokenizer,  # IDSTokenizer instance
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
            tokenizer: IDS tokenizer for text processing
            train_dataloader: Training data loader
            val_dataloader: Validation data loader  
            optimizer: Optimizer for trainable parameters
            lr_scheduler: Learning rate scheduler
            device: Training device (cuda/cpu)
            scheduler: Diffusers noise scheduler for diffusion process
        """
        self.config = config
        self.models = models
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.scheduler = scheduler
        
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
    
    def train(self):
        """The main training loop."""
        print("Starting Stage 1 training...")
        
        for epoch in range(self.config['stage1']['num_train_epochs']):
            print(f"--- Epoch {epoch+1}/{self.config['stage1']['num_train_epochs']} ---")
            
            for step, batch in enumerate(self.train_dataloader):
                with torch.cuda.amp.autocast(enabled=self.config['stage1']['use_amp']):
                    loss = self._train_step(batch)
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
            latents = self.models['vae'].encode(batch['pixel_values']).latent_dist.sample()
            latents = latents * scaling_factor
        
        # 2. Sample noise and timesteps for diffusion process
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, 
            (latents.shape[0],), 
            device=self.device
        ).long()
        
        # 3. Add noise to latents (forward diffusion process)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # 4. Get text conditioning from our new IDS-based modules
        try:
            # Use the new tokenized batch processing method
            text_embeds = self.models['ids_text_embedder'].forward_tokenized_batch(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch['token_type_ids'],
                text_pos=batch['text_pos']
            )  # (batch_size, 112, 128) - token-level features
            
            # Transform through adapter to get conditioning features
            text_features = self.models['adapter'](text_embeds)  # (batch_size, 112, projection_dim)
            
        except Exception as e:
            logger.error(f"Error in text processing: {e}")
            # Fallback to zero conditioning
            text_features = torch.zeros(
                latents.shape[0], 112, 4096, 
                device=self.device, dtype=latents.dtype
            )
        
        # 5. Get ControlNet guidance from TextRenderNet
        try:
            with torch.no_grad():
                # Prepare conditioning images (masked images)
                conditioning_images = batch['conditioning_pixel_values']
                
                # Get ControlNet residuals
                controlnet_output = self.models['text_render_net'](
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=text_features,
                    controlnet_cond=conditioning_images,
                    return_dict=False
                )
                
        except Exception as e:
            logger.error(f"Error in ControlNet processing: {e}")
            # Fallback to no control guidance
            down_block_res_samples = None
            mid_block_res_sample = None
        
        # 6. Predict noise with the main SD3 Transformer
        try:
            model_pred = self.models['transformer'](
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=text_features,
                pooled_projections=None,  # Not used in this stage
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
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
            # Use the tokenized batch processing method
            text_embeds = self.models['ids_text_embedder'].forward_tokenized_batch(
                input_ids=val_batch['input_ids'],
                attention_mask=val_batch['attention_mask'],
                token_type_ids=val_batch['token_type_ids'],
                text_pos=val_batch['text_pos']
            )  # (batch_size, 112, 128) - token-level features
            
            # Transform through adapter to get conditioning features
            text_features = self.models['adapter'](text_embeds)  # (batch_size, 112, projection_dim)
            
            # Prepare ControlNet conditioning
            controlnet_cond = val_batch['conditioning_pixel_values']  # (batch_size, 3, H, W)

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
                    # Get ControlNet guidance
                    down_block_res_samples, mid_block_res_sample = self.models['text_render_net'](
                        sample=latents,
                        timestep=timestep,
                        encoder_hidden_states=text_features,
                        controlnet_cond=controlnet_cond,
                        return_dict=False
                    )
                    
                    # Predict noise with main transformer
                    noise_pred = self.models['transformer'](
                        hidden_states=latents,
                        timestep=timestep,
                        encoder_hidden_states=text_features,
                        pooled_projections=None,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
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
            conditioning_save_path = os.path.join(output_dir, f"conditioning_epoch_{epoch}_step_{step}.png")
            conditioning_images = (controlnet_cond / 2 + 0.5).clamp(0, 1)  # Denormalize
            save_image(conditioning_images, conditioning_save_path, nrow=2, normalize=False)
            
            # Save original images for reference
            original_save_path = os.path.join(output_dir, f"original_epoch_{epoch}_step_{step}.png")
            original_images = (val_batch['pixel_values'] / 2 + 0.5).clamp(0, 1)  # Denormalize
            save_image(original_images, original_save_path, nrow=2, normalize=False)
            
            print(f"✓ Validation images saved:")
            print(f"  Generated: {save_path}")
            print(f"  Conditioning: {conditioning_save_path}")
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