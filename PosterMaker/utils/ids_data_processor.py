"""
IDS-based Data Processor for PosterMaker
Optimized for LoRA training with IDS structural decomposition
"""

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.ids_text_embedder import IDSTextEmbedder
from utils.utils import *


class IDSDataProcessor:
    """
    Enhanced data processor with IDS-based text embedding
    Designed for LoRA training and architectural improvements
    """
    
    def __init__(self, input_size=(1024, 1024), erode_mask=False, 
                 vocab_file='./assets/ids_vocab.json', 
                 use_structural_attention=True,
                 training_mode=True):
        """
        Args:
            input_size: Input image size (H, W)
            erode_mask: Whether to apply erosion to subject mask
            vocab_file: Path to IDS vocabulary file
            use_structural_attention: Whether to use structural attention in IDS embedder
            training_mode: Whether in training mode (affects dropout, etc.)
        """
        self.input_size = input_size
        self.erode_mask = erode_mask
        self.training_mode = training_mode
        
        # Use IDS text embedder with optimized settings
        self.text_embedder = IDSTextEmbedder(
            vocab_file=vocab_file,
            ids_embed_dim=64,  # Fixed to match original architecture
            max_seq_length=128,
            max_num_texts=7,
            max_chars_per_text=16,
            input_size=input_size,
            use_structural_attention=use_structural_attention
        )
        
        # Set training mode
        if training_mode:
            self.text_embedder.train()
        else:
            self.text_embedder.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])
        ])

    def set_training_mode(self, training: bool):
        """Set training mode for the text embedder"""
        self.training_mode = training
        if training:
            self.text_embedder.train()
        else:
            self.text_embedder.eval()
    
    def get_lora_target_modules(self):
        """Get LoRA target modules from the text embedder"""
        return self.text_embedder.get_lora_target_modules()
    
    def __call__(self, image, mask, texts, prompt, return_intermediate=False):
        """
        Process input with IDS-based text embedding
        
        Args:
            image: Input image (H, W, C) 
            mask: Subject mask for inpainting
            texts: List of text dictionaries with 'content' and 'pos'
            prompt: Text prompt for generation
            return_intermediate: Whether to return intermediate features for analysis
        
        Returns:
            Dictionary with processed data ready for ControlNet training/inference
        """

        # Standard image preprocessing (same as original)
        image = convert_to_rgb(image)
        input_size = self.input_size
        poster_h, poster_w, _ = image.shape
        new_h, new_w, resize_scale = cal_resize_and_padding((poster_h, poster_w), input_size)
        
        processed_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        subject_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        if self.erode_mask:
            subject_mask = cv2.erode(subject_mask, np.ones((3, 3), np.uint8), iterations=1)

        # Adjust text positions according to resize
        adjusted_texts = []
        for text in texts:
            adjusted_text = text.copy()
            adjusted_text['pos'] = reisize_box_by_scale(text['pos'], resize_scale)
            adjusted_text['pos'] = clamp_bbox_to_image(adjusted_text['pos'], new_w, new_h)
            adjusted_texts.append(adjusted_text)
        
        # Create text mask
        text_mask = create_mask_by_text((new_h, new_w), adjusted_texts)
        empty_image = np.zeros_like(processed_image)

        # Padding to input_size
        processed_image = pad_image_to_shape(processed_image, input_size, pad_value=255)
        text_mask = pad_image_to_shape(text_mask, input_size, pad_value=255)
        subject_mask = pad_image_to_shape(subject_mask, input_size, pad_value=255)
        empty_image = pad_image_to_shape(empty_image, input_size, pad_value=0)
        
        # IDS-based text embedding
        # Note: No torch.no_grad() during training to allow gradient flow
        if self.training_mode:
            text_embeds = self.text_embedder(adjusted_texts)  # (112, 128)
        else:
        with torch.no_grad():
                text_embeds = self.text_embedder(adjusted_texts)  # (112, 128)

        # Tensor transformation and normalization
        processed_image = self.transform(processed_image)
        subject_mask = self.transform(subject_mask)
        text_mask = self.transform(text_mask)
        empty_image = self.transform(empty_image)

        # SD3 inpaint controlnet input preparation
        control_mask = subject_mask
        control_mask = ((control_mask + 1.0) / 2.0)  # [-1,1]->[0,1], 0 means need inpaint
        cond_image_inpaint = (processed_image + 1) * control_mask - 1

        # Add batch dimension
        cond_image_inpaint = cond_image_inpaint.unsqueeze(0)
        control_mask = control_mask.unsqueeze(0)
        text_embeds = text_embeds.unsqueeze(0)  # (1, 112, 128)
        empty_image = empty_image.unsqueeze(0)
        
        # Prepare return data
        result = {
            'cond_image_inpaint': cond_image_inpaint,
            'control_mask': control_mask,
            'prompt': prompt,
            'text_embeds': text_embeds,  # (1, 112, 128) - correct shape for adapter
            'target_size': (new_h, new_w),
            'controlnet_im': empty_image,
            'adjusted_texts': adjusted_texts  # Include adjusted text info
        }
        
        # Add intermediate features for debugging/analysis if requested
        if return_intermediate:
            result.update({
                'original_texts': texts,
                'resize_scale': resize_scale,
                'text_mask': text_mask.unsqueeze(0),
                'processed_image': processed_image.unsqueeze(0)
            })
        
        return result


class IDSTrainingDataProcessor(IDSDataProcessor):
    """
    Extended data processor specifically designed for LoRA training
    Includes additional features for training monitoring and optimization
    """
    
    def __init__(self, input_size=(1024, 1024), erode_mask=False, 
                 vocab_file='./assets/ids_vocab.json',
                 use_structural_attention=True,
                 enable_data_augmentation=True):
        """
        Args:
            enable_data_augmentation: Whether to apply data augmentation during training
        """
        super().__init__(
            input_size=input_size,
            erode_mask=erode_mask, 
            vocab_file=vocab_file,
            use_structural_attention=use_structural_attention,
            training_mode=True
        )
        
        self.enable_data_augmentation = enable_data_augmentation
        
        # Data augmentation for text positions (optional)
        self.position_jitter_range = 0.02  # 2% of image size
        
    def _apply_text_position_augmentation(self, texts, image_size):
        """
        Apply slight random jitter to text positions for data augmentation
        
        Args:
            texts: List of text dictionaries
            image_size: (height, width) of the image
            
        Returns:
            Augmented texts with slightly modified positions
        """
        if not self.enable_data_augmentation or not self.training_mode:
            return texts
            
        augmented_texts = []
        h, w = image_size
        
        for text in texts:
            augmented_text = text.copy()
            pos = text['pos']  # [x1, y1, x2, y2]
            
            # Calculate jitter amounts
            jitter_x = np.random.uniform(-self.position_jitter_range, self.position_jitter_range) * w
            jitter_y = np.random.uniform(-self.position_jitter_range, self.position_jitter_range) * h
            
            # Apply jitter while maintaining box size
            new_pos = [
                max(0, min(w - (pos[2] - pos[0]), pos[0] + jitter_x)),
                max(0, min(h - (pos[3] - pos[1]), pos[1] + jitter_y)),
                0,  # Will be calculated below
                0   # Will be calculated below
            ]
            new_pos[2] = new_pos[0] + (pos[2] - pos[0])  # x2
            new_pos[3] = new_pos[1] + (pos[3] - pos[1])  # y2
            
            # Ensure within image bounds
            new_pos[2] = min(w, new_pos[2])
            new_pos[3] = min(h, new_pos[3])
            
            augmented_text['pos'] = new_pos
            augmented_texts.append(augmented_text)
            
        return augmented_texts
    
    def get_training_statistics(self):
        """
        Get statistics useful for monitoring LoRA training
        
        Returns:
            Dictionary with training-relevant statistics
        """
        stats = {
            'vocab_size': self.text_embedder.tokenizer.vocab_size,
            'max_seq_length': self.text_embedder.ids_embedding.max_seq_length,
            'embedding_dim': self.text_embedder.ids_embedding.embed_dim,
            'use_structural_attention': self.text_embedder.use_structural_attention,
            'lora_target_modules': self.get_lora_target_modules()
        }
        
        return stats
    
    def __call__(self, image, mask, texts, prompt, 
                 apply_augmentation=None, return_training_info=False):
        """
        Enhanced call method for training with optional augmentation
        
        Args:
            apply_augmentation: Override the default augmentation setting
            return_training_info: Whether to return additional training information
        """
        
        # Apply text position augmentation if enabled
        original_augmentation = self.enable_data_augmentation
        if apply_augmentation is not None:
            self.enable_data_augmentation = apply_augmentation
            
        poster_h, poster_w = image.shape[:2]
        augmented_texts = self._apply_text_position_augmentation(texts, (poster_h, poster_w))
        
        # Restore original augmentation setting
        self.enable_data_augmentation = original_augmentation
        
        # Call parent processing method
        result = super().__call__(
            image, mask, augmented_texts, prompt, 
            return_intermediate=return_training_info
        )
        
        # Add training-specific information
        if return_training_info:
            result.update({
                'training_stats': self.get_training_statistics(),
                'augmentation_applied': apply_augmentation if apply_augmentation is not None else original_augmentation,
                'original_text_positions': [text['pos'] for text in texts],
                'augmented_text_positions': [text['pos'] for text in augmented_texts]
            })
        
        return result


# Backward compatibility alias
IDSUserInputProcessor = IDSDataProcessor 