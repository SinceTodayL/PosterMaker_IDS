"""
Dataset implementation for Stage 1 training of PosterMaker IDS-based pipeline.
This dataset processes ALL valid text annotations from full poster images.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional

from .utils.ids_tokenizer import IDSTokenizer
from .utils.text_utils import create_text_mask, mask_image_region, validate_text_annotation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PosterDatasetStage1(Dataset):
    """
    PyTorch Dataset for Stage 1 training of PosterMaker.
    This dataset processes ALL valid text annotations from full poster images.
    """
    
    def __init__(self, config: Dict[str, Any], tokenizer: IDSTokenizer, split: str = 'train'):
        """
        Initialize the dataset.
        
        Args:
            config (Dict[str, Any]): The loaded training configuration
            tokenizer (IDSTokenizer): An instance of our IDSTokenizer
            split (str): The dataset split, 'train' or 'val'
        """
        # Store config and tokenizer
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        
        # Construct the absolute path to the dataset split directory
        self.data_root = os.path.join(config['dataset_dir'], split)
        self.prompt_key = config.get('prompt_key', 'prompt')
        
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Dataset split directory not found: {self.data_root}")
        
        # Get a list of all sample directories
        self.samples = [d for d in os.listdir(self.data_root) 
                       if os.path.isdir(os.path.join(self.data_root, d))]
        
        if not self.samples:
            raise ValueError(f"No samples found in {self.data_root}")
        
        logger.info(f"Loaded {len(self.samples)} samples from {split} split")
        
        # Target size for training (keep as 1024 for compatibility)
        self.target_size = 1024
        
        # Define image transformations with aspect ratio preservation
        self.image_transforms = transforms.Compose([
            transforms.Lambda(lambda img: self._resize_with_padding(img, self.target_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])  # Normalize to [-1, 1]
        ])
        
        # Define mask transformations (no normalization for masks)
        self.mask_transforms = transforms.Compose([
            transforms.Lambda(lambda img: self._resize_with_padding(img, self.target_size, is_mask=True)),
            transforms.ToTensor()
        ])
        
        # Stage 1 specific parameters
        self.max_seq_length = config['stage1']['max_seq_length']
        self.mask_fill_value = 0.0  # Value to fill masked text regions
        
        # Statistics for monitoring
        self.stats = {
            'total_samples': len(self.samples),
            'valid_annotations': 0,
            'empty_text_samples': 0,
            'error_samples': 0
        }
        
        # Validate samples (optional, can be disabled for faster startup)
        if split == 'train':
            self._validate_samples()
    
    def _validate_samples(self):
        """Validate a subset of samples to ensure data quality"""
        logger.info("Validating sample data quality...")
        
        # Validate first 100 samples or all samples if fewer than 100
        samples_to_check = min(100, len(self.samples))
        
        for i in range(samples_to_check):
            sample_dir = os.path.join(self.data_root, self.samples[i])
            annotation_path = os.path.join(sample_dir, 'annotation.json')
            
            try:
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    annotation = json.load(f)
                
                if validate_text_annotation(annotation):
                    self.stats['valid_annotations'] += 1
                    if not annotation['texts']:
                        self.stats['empty_text_samples'] += 1
                    if self.prompt_key not in annotation or not annotation[self.prompt_key]:
                        logger.warning(f"Prompt key '{self.prompt_key}' not found or empty in sample: {self.samples[i]}")
                else:
                    logger.warning(f"Invalid annotation in sample: {self.samples[i]}")
                    
            except Exception as e:
                logger.error(f"Error validating sample {self.samples[i]}: {e}")
                self.stats['error_samples'] += 1
        
        logger.info(f"Validation complete: {self.stats['valid_annotations']}/{samples_to_check} valid samples")
    
    def _resize_with_padding(self, image: Image.Image, target_size: int, is_mask: bool = False) -> Image.Image:
        """
        Resize image while preserving aspect ratio and pad to square (following PosterMaker paper).
        Images are aligned to top-left corner and padded with white background.
        
        Args:
            image: PIL Image to resize
            target_size: Target size (e.g., 1024 for 1024x1024)
            is_mask: Whether this is a mask image (affects padding color)
            
        Returns:
            PIL Image resized and padded to target_size x target_size
        """
        width, height = image.size
        
        # Calculate scale following PosterMaker method: min(target_h/ori_h, target_w/ori_w)
        scale = min(target_size / height, target_size / width)
        new_width, new_height = int(width * scale), int(height * scale)
        
        # Resize maintaining aspect ratio
        resized = image.resize((new_width, new_height), Image.BILINEAR if not is_mask else Image.NEAREST)
        
        # Create target_size x target_size background (top-left alignment)
        if is_mask:
            # For masks, use white background (255) to match PosterMaker
            padded = Image.new(image.mode, (target_size, target_size), 255)
        else:
            # For images, use white background to match PosterMaker
            padded = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        
        # Paste at top-left corner (0, 0) - following PosterMaker paper
        offset = (0, 0)
        padded.paste(resized, offset)
        
        # Store padding info for coordinate adjustment (simplified - no offset needed)
        self._last_padding_info = {
            'scale': scale,
            'offset': offset,  # Always (0,0) now
            'original_size': (width, height),
            'new_size': (new_width, new_height)
        }
        
        # Log the transformation for debugging (only occasionally to avoid spam)
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1
            
        if self._debug_count <= 3:  # Only log first 3 transformations
            logger.info(f"Image resize: {width}x{height} → {new_width}x{new_height} (scale={scale:.3f}) → {target_size}x{target_size} (offset={offset})")
        
        return padded
    
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.samples)
    
    def _load_sample_data(self, sample_dir: str) -> Tuple[Dict, Image.Image, Optional[Image.Image]]:
        """
        Load annotation, image, and optionally subject mask for a sample.
        
        Args:
            sample_dir: Path to sample directory
            
        Returns:
            Tuple of (annotation_dict, image, subject_mask)
        """
        # Load annotation
        annotation_path = os.path.join(sample_dir, 'annotation.json')
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)
        
        # Load main image (support multiple formats)
        image_path = None
        for ext in ['png', 'jpg', 'jpeg']:
            candidate_path = os.path.join(sample_dir, f'image.{ext}')
            if os.path.exists(candidate_path):
                image_path = candidate_path
                break
        
        if image_path is None:
            raise FileNotFoundError(f"Image file not found in {sample_dir}. Supported formats: png, jpg, jpeg")
        
        image = Image.open(image_path).convert("RGB")
        
        # Load subject mask if available
        subject_mask_path = os.path.join(sample_dir, 'subject_mask.png')
        subject_mask = None
        if os.path.exists(subject_mask_path):
            subject_mask = Image.open(subject_mask_path).convert("L")
        
        return annotation, image, subject_mask
    
    def _create_text_conditioning_mask(self, text_pos: List[int], image_size: Tuple[int, int]) -> Image.Image:
        """
        Create conditioning mask for the selected text region.
        
        Args:
            text_pos: Text position [x1, y1, x2, y2]
            image_size: (width, height) of the image
            
        Returns:
            PIL Image mask with the text region marked as 1 (white)
        """
        width, height = image_size
        mask_array = torch.zeros(height, width)
        
        x1, y1, x2, y2 = text_pos
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        if x2 > x1 and y2 > y1:
            mask_array[y1:y2, x1:x2] = 1.0
        
        # Convert to PIL Image
        mask_pil = Image.fromarray((mask_array.numpy() * 255).astype('uint8'), mode='L')
        return mask_pil
    
    def _create_conditioning_image(self, image: Image.Image, text_positions: List[List[int]]) -> Image.Image:
        """
        Create conditioning image by masking the text regions.
        
        Args:
            image (Image.Image): Original PIL image.
            text_positions (List[List[int]]): List of text positions [[x1, y1, x2, y2], ...].
            
        Returns:
            Image.Image: PIL image with text regions masked.
        """
        # Apply the same transformation that will be applied to the main image
        image_tensor = self.image_transforms(image)
        
        # Return original transformed image if no text to mask
        if not text_positions:
            # Convert back to PIL
            conditioning_array = ((image_tensor + 1) / 2).permute(1, 2, 0).numpy()
            conditioning_array = (conditioning_array * 255).astype('uint8')
            return Image.fromarray(conditioning_array, mode='RGB')

        # Calculate adjusted positions for the padded/scaled image (top-left alignment)
        orig_width, orig_height = image.size
        scale = min(self.target_size / orig_height, self.target_size / orig_width)
        offset_x, offset_y = 0, 0  # No offset needed for top-left alignment
        
        # Adjust text positions
        adjusted_positions = []
        for pos in text_positions:
            x1, y1, x2, y2 = pos
            adjusted_x1 = x1 * scale + offset_x
            adjusted_y1 = y1 * scale + offset_y
            adjusted_x2 = x2 * scale + offset_x
            adjusted_y2 = y2 * scale + offset_y
            adjusted_positions.append([int(adjusted_x1), int(adjusted_y1), int(adjusted_x2), int(adjusted_y2)])
        
        # Create mask for the adjusted text regions
        mask = create_text_mask(adjusted_positions, (self.target_size, self.target_size))
        
        # Mask the image (fill text region with neutral gray value 0.0 in [-1,1] range)
        conditioning_image_tensor = mask_image_region(image_tensor, mask, fill_value=self.mask_fill_value)
        
        # Convert back to PIL for consistency
        # First convert from [-1,1] to [0,1]
        conditioning_array = ((conditioning_image_tensor + 1) / 2).permute(1, 2, 0).numpy()
        conditioning_array = (conditioning_array * 255).astype('uint8')
        conditioning_image = Image.fromarray(conditioning_array, mode='RGB')
        
        return conditioning_image
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load a sample and process ALL valid text annotations.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing training data for Stage 1 with all texts
        """
        original_idx = idx
        max_retries = 5  # Prevent infinite loops on bad data
        
        for retry in range(max_retries):
            try:
                sample_dir = os.path.join(self.data_root, self.samples[idx])
                
                # Load sample data
                annotation, image, subject_mask = self._load_sample_data(sample_dir)
                
                # --- Process ALL Text Annotations ---
                
                # 1. Check if there are text annotations
                if not annotation.get('texts') or len(annotation['texts']) == 0:
                    # No text annotations, try next sample sequentially
                    idx = (original_idx + retry + 1) % len(self)
                    continue
                
                # 2. Validate and collect ALL text boxes
                valid_texts = []
                for text_box in annotation['texts']:
                    content = text_box.get('content', '')
                    pos = text_box.get('pos', [])
                    
                    # Validate text content
                    if not content or not content.strip():
                        continue  # Skip this text, but don't abandon the whole sample
                    
                    # Validate position
                    if len(pos) != 4 or pos[2] <= pos[0] or pos[3] <= pos[1]:
                        continue  # Skip this text, but don't abandon the whole sample
                    
                    valid_texts.append({
                        'content': content.strip(),
                        'pos': pos
                    })
                
                # Check if we have any valid texts
                if not valid_texts:
                    # No valid texts, try next sample sequentially
                    idx = (original_idx + retry + 1) % len(self)
                    continue
                
                # 3. Create conditioning masks and images for ALL texts
                all_text_positions = [t['pos'] for t in valid_texts]
                
                # For SceneGenNet: mask ALL text regions
                scene_conditioning_image = self._create_conditioning_image(image, all_text_positions)
                
                # For TextRenderNet: also mask ALL text regions (will be rendered by our model)
                text_render_conditioning_image = self._create_conditioning_image(image, all_text_positions)
                
                # Create combined conditioning mask for all text regions (adjusted for padding)
                combined_mask = torch.zeros(self.target_size, self.target_size)  # H x W, using target size
                
                # Get padding info for coordinate adjustment
                orig_width, orig_height = image.size
                scale = min(self.target_size / orig_height, self.target_size / orig_width)
                
                for text_pos in all_text_positions:
                    x1, y1, x2, y2 = text_pos
                    
                    # Apply scaling only (no offset needed for top-left alignment)
                    scaled_x1 = int(x1 * scale)
                    scaled_y1 = int(y1 * scale)
                    scaled_x2 = int(x2 * scale)
                    scaled_y2 = int(y2 * scale)
                    
                    # Ensure coordinates are within target size bounds
                    scaled_x1 = max(0, min(scaled_x1, self.target_size))
                    scaled_y1 = max(0, min(scaled_y1, self.target_size))
                    scaled_x2 = max(0, min(scaled_x2, self.target_size))
                    scaled_y2 = max(0, min(scaled_y2, self.target_size))
                    
                    if scaled_x2 > scaled_x1 and scaled_y2 > scaled_y1:
                        combined_mask[scaled_y1:scaled_y2, scaled_x1:scaled_x2] = 1.0
                
                conditioning_mask_pil = Image.fromarray((combined_mask.numpy() * 255).astype('uint8'), mode='L')
                
                # 4. Prepare text data - combine all texts for processing
                combined_content = ' '.join([t['content'] for t in valid_texts])
                
                # 5. Tokenize the combined text content
                tokenized_text = self.tokenizer.encode_text(
                    combined_content,
                    max_length=self.max_seq_length,
                    add_special_tokens=True,
                    use_recursive=False,  # Simple decomposition for training efficiency
                    enhance_rare_chars=False  # No complex enhancement for training
                )
                
                # 6. Apply transformations to images
                pixel_values = self.image_transforms(image)
                scene_conditioning_pixel_values = self.image_transforms(scene_conditioning_image)
                text_render_conditioning_pixel_values = self.image_transforms(text_render_conditioning_image)
                conditioning_mask = self.mask_transforms(conditioning_mask_pil)
                
                # 7. Get the prompt
                prompt = annotation.get(self.prompt_key, "")
                if not prompt:
                    logger.debug(f"Sample {self.samples[idx]} has an empty prompt.")

                # 8. Prepare all text positions and info (adjusted for padding and scaling)
                orig_width, orig_height = image.size
                
                # Get padding info from the last resize operation
                if hasattr(self, '_last_padding_info'):
                    padding_info = self._last_padding_info
                    scale = padding_info['scale']
                    offset_x, offset_y = padding_info['offset']
                else:
                    # Fallback if padding info is not available (top-left alignment)
                    scale = min(self.target_size / orig_height, self.target_size / orig_width)
                    offset_x, offset_y = 0, 0  # No offset for top-left alignment
                
                all_texts_info = []
                for text_info in valid_texts:
                    # Original coordinates
                    x1, y1, x2, y2 = text_info['pos']
                    
                    # Apply scaling and offset to match the padded image
                    scaled_x1 = x1 * scale + offset_x
                    scaled_y1 = y1 * scale + offset_y
                    scaled_x2 = x2 * scale + offset_x
                    scaled_y2 = y2 * scale + offset_y
                    
                    # Normalize to [0, 1] based on target size (1024x1024)
                    text_pos_norm = [
                        scaled_x1 / self.target_size,
                        scaled_y1 / self.target_size,
                        scaled_x2 / self.target_size,
                        scaled_y2 / self.target_size
                    ]
                    
                    all_texts_info.append({
                        'content': text_info['content'],
                        'pos': text_pos_norm
                    })
                
                # Use first text position for backward compatibility
                primary_text_pos = all_texts_info[0]['pos'] if all_texts_info else [0.0, 0.0, 1.0, 1.0]
                
                return {
                    "pixel_values": pixel_values,
                    "scene_conditioning_pixel_values": scene_conditioning_pixel_values,
                    "text_render_conditioning_pixel_values": text_render_conditioning_pixel_values,
                    "conditioning_mask": conditioning_mask,
                    "input_ids": torch.tensor(tokenized_text['input_ids'], dtype=torch.long),
                    "attention_mask": torch.tensor(tokenized_text['attention_mask'], dtype=torch.long), 
                    "token_type_ids": torch.tensor(tokenized_text['token_type_ids'], dtype=torch.long),
                    "text_pos": torch.tensor(primary_text_pos, dtype=torch.float32),
                    "prompt": prompt,
                    "text_content": combined_content,
                    "all_texts_info": all_texts_info,
                    "sample_id": self.samples[idx]
                }
                
            except Exception as e:
                logger.warning(f"Error loading sample {self.samples[idx]} (attempt {retry + 1}): {e}")
                # Try next sample sequentially to avoid infinite loops
                idx = (original_idx + retry + 1) % len(self)
                
                if retry == max_retries - 1:
                    # Last retry failed, raise exception to stop training
                    logger.error(f"Failed to load sample after {max_retries} retries")
                    raise RuntimeError(f"Cannot load valid data after {max_retries} retries")
        
        # This should never be reached, but just in case
        raise RuntimeError("Dataset loading failed unexpectedly")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return {
            **self.stats,
            'split': self.split,
            'data_root': self.data_root,
            'total_samples': len(self.samples),
            'tokenizer_vocab_size': self.tokenizer.vocab_size
        }
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for batching samples with multiple texts support.
        
        Args:
            batch: List of samples from __getitem__
            
        Returns:
            Batched data dictionary
        """
        # Stack image tensors
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        scene_conditioning_pixel_values = torch.stack([item["scene_conditioning_pixel_values"] for item in batch])
        text_render_conditioning_pixel_values = torch.stack([item["text_render_conditioning_pixel_values"] for item in batch])
        conditioning_mask = torch.stack([item["conditioning_mask"] for item in batch])
        
        # Stack text data (already padded by tokenizer)
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        token_type_ids = torch.stack([item["token_type_ids"] for item in batch])
        
        # Stack position data
        text_pos = torch.stack([item["text_pos"] for item in batch])
        
        # Keep prompt, text content and sample IDs as lists
        prompts = [item["prompt"] for item in batch]
        text_content = [item["text_content"] for item in batch]
        sample_ids = [item["sample_id"] for item in batch]
        
        # Handle all texts info
        all_texts_info = [item["all_texts_info"] for item in batch]
        
        return {
            "pixel_values": pixel_values,
            "scene_conditioning_pixel_values": scene_conditioning_pixel_values,
            "text_render_conditioning_pixel_values": text_render_conditioning_pixel_values,
            "conditioning_mask": conditioning_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "text_pos": text_pos,
            "prompt": prompts,
            "text_content": text_content,
            "sample_ids": sample_ids,
            "all_texts_info": all_texts_info
        }


def create_stage1_dataloader(config: Dict[str, Any], 
                           tokenizer: IDSTokenizer, 
                           split: str = 'train',
                           **dataloader_kwargs) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for Stage 1 training.
    
    Args:
        config: Training configuration
        tokenizer: IDS tokenizer instance
        split: Dataset split ('train' or 'val')
        **dataloader_kwargs: Additional arguments for DataLoader
        
    Returns:
        Configured DataLoader
    """
    dataset = PosterDatasetStage1(config, tokenizer, split)
    
    # Default DataLoader settings for Stage 1
    default_kwargs = {
        'batch_size': config['stage1']['batch_size'],
        'shuffle': (split == 'train'),
        'num_workers': 2,
        'pin_memory': True,
        'collate_fn': dataset.collate_fn,
        'drop_last': (split == 'train')  # Drop last incomplete batch for training
    }
    
    # Override with provided kwargs
    default_kwargs.update(dataloader_kwargs)
    
    dataloader = torch.utils.data.DataLoader(dataset, **default_kwargs)
    
    logger.info(f"Created {split} dataloader with {len(dataset)} samples, "
               f"batch_size={default_kwargs['batch_size']}")
    
    return dataloader
