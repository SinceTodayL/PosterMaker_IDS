"""
IDS-based Data Processor for PosterMaker
Supports IDS text embedding instead of character-level features
"""

import cv2
import torch.nn.functional as F
from torchvision import transforms
from models.ids_text_embedder import IDSTextEmbedder
from utils.utils import *


class IDSUserInputProcessor:
    """
    User input processor with IDS-based text embedding
    Replaces character-level TextEmbedder with IDS structural decomposition
    """
    
    def __init__(self, input_size=(1024, 1024), erode_mask=False, 
                 vocab_file='./assets/ids_vocab.json'):
        self.input_size = input_size
        self.erode_mask = erode_mask
        
        # Use IDS text embedder instead of character-level
        self.text_embedder = IDSTextEmbedder(
            vocab_file=vocab_file,
            max_num_texts=7,
            max_chars_per_text=16,
            input_size=input_size
        )
        
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])
        ])

    def __call__(self, image, mask, texts, prompt):
        """
        Preprocess user input with IDS text embedding
        
        Parameters:
        image: numpy array, Input image (H, W, C)
        mask: numpy array, Subject mask (H, W)
        texts: list of dict, Text content and location information
               Each dict format: {"content": str, "pos": [x1,y1,x2,y2]}
        prompt: str, Text prompt for scene generation
        
        Returns:
        dict: Preprocessed data with IDS text embeddings
        """

        # Convert RGBA to RGB if needed
        image = convert_to_rgb(image)

        # Resize and preprocessing
        input_size = self.input_size
        poster_h, poster_w, _ = image.shape
        new_h, new_w, resize_scale = cal_resize_and_padding((poster_h, poster_w), input_size)
        
        processed_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        subject_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        if self.erode_mask:
            subject_mask = cv2.erode(subject_mask, np.ones((3, 3), np.uint8), iterations=1)

        # Adjust text box positions for resizing
        for text in texts:
            text['pos'] = reisize_box_by_scale(text['pos'], resize_scale)
            text['pos'] = clamp_bbox_to_image(text['pos'], new_w, new_h)
        
        # Create text mask (for visualization/debugging)
        text_mask = create_mask_by_text((new_h, new_w), texts)

        # Create empty image for control
        empty_image = np.zeros_like(processed_image)

        # Pad to input_size
        processed_image = pad_image_to_shape(processed_image, input_size, pad_value=255)
        text_mask = pad_image_to_shape(text_mask, input_size, pad_value=255)
        subject_mask = pad_image_to_shape(subject_mask, input_size, pad_value=255)
        empty_image = pad_image_to_shape(empty_image, input_size, pad_value=0)
        
        # IDS text embeddings - this is the key difference
        with torch.no_grad():
            text_embeds = self.text_embedder(texts)  # (max_total_tokens, feature_dim)

        # Tensor conversion and normalization
        processed_image = self.transform(processed_image)
        subject_mask = self.transform(subject_mask)
        text_mask = self.transform(text_mask)
        empty_image = self.transform(empty_image)

        # SD3 inpaint controlnet input preparation
        control_mask = subject_mask
        control_mask = ((control_mask + 1.0) / 2.0)  # [-1,1] -> [0,1], 0 means need inpaint
        cond_image_inpaint = (processed_image + 1) * control_mask - 1

        # Add batch dimension: 1, C, H, W
        cond_image_inpaint = cond_image_inpaint.unsqueeze(0)
        control_mask = control_mask.unsqueeze(0)
        text_embeds = text_embeds.unsqueeze(0)  # (1, max_total_tokens, feature_dim)
        empty_image = empty_image.unsqueeze(0)
        
        # Return preprocessed data
        result = {
            'cond_image_inpaint': cond_image_inpaint,
            'control_mask': control_mask,
            'prompt': prompt,
            'text_embeds': text_embeds,  # IDS-based embeddings
            'target_size': (new_h, new_w),
            'controlnet_im': empty_image,
            # Optional: keep original data for debugging
            'texts': texts,
            'text_mask': text_mask.unsqueeze(0),
            'original_size': (poster_h, poster_w),
        }
        
        return result


class IDSDataProcessor:
    """
    Enhanced data processor with both character-level and IDS support
    Allows switching between different text embedding methods
    """
    
    def __init__(self, input_size=(1024, 1024), erode_mask=False, 
                 vocab_file='./assets/ids_vocab.json', use_ids=True):
        self.input_size = input_size
        self.erode_mask = erode_mask
        self.use_ids = use_ids
        
        if use_ids:
            # Use IDS text embedder
            self.text_embedder = IDSTextEmbedder(
                vocab_file=vocab_file,
                max_num_texts=7,
                max_chars_per_text=16,
                input_size=input_size
            )
        else:
            # Fall back to original character-level embedder
            from models.text_embedder import TextEmbedder
            self.text_embedder = TextEmbedder()
            
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])
        ])
        
    def switch_embedding_method(self, use_ids: bool):
        """Switch between IDS and character-level embedding"""
        if use_ids != self.use_ids:
            self.use_ids = use_ids
            if use_ids:
                self.text_embedder = IDSTextEmbedder(
                    vocab_file='./assets/ids_vocab.json',
                    max_num_texts=7,
                    max_chars_per_text=16,
                    input_size=self.input_size
                )
            else:
                from models.text_embedder import TextEmbedder
                self.text_embedder = TextEmbedder()
    
    def __call__(self, image, mask, texts, prompt):
        """Process input with selected embedding method"""
        
        # Standard image preprocessing (same for both methods)
        image = convert_to_rgb(image)
        input_size = self.input_size
        poster_h, poster_w, _ = image.shape
        new_h, new_w, resize_scale = cal_resize_and_padding((poster_h, poster_w), input_size)
        
        processed_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        subject_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        if self.erode_mask:
            subject_mask = cv2.erode(subject_mask, np.ones((3, 3), np.uint8), iterations=1)

        # Adjust text positions
        for text in texts:
            text['pos'] = reisize_box_by_scale(text['pos'], resize_scale)
            text['pos'] = clamp_bbox_to_image(text['pos'], new_w, new_h)
        
        text_mask = create_mask_by_text((new_h, new_w), texts)
        empty_image = np.zeros_like(processed_image)

        # Padding
        processed_image = pad_image_to_shape(processed_image, input_size, pad_value=255)
        text_mask = pad_image_to_shape(text_mask, input_size, pad_value=255)
        subject_mask = pad_image_to_shape(subject_mask, input_size, pad_value=255)
        empty_image = pad_image_to_shape(empty_image, input_size, pad_value=0)
        
        # Text embedding - this differs based on method
        if self.use_ids:
            # IDS-based embedding
            with torch.no_grad():
                text_embeds = self.text_embedder(texts)
        else:
            # Character-level embedding
            text_embeds = self.text_embedder(texts)

        # Standard tensor processing
        processed_image = self.transform(processed_image)
        subject_mask = self.transform(subject_mask)
        text_mask = self.transform(text_mask)
        empty_image = self.transform(empty_image)

        control_mask = subject_mask
        control_mask = ((control_mask + 1.0) / 2.0)
        cond_image_inpaint = (processed_image + 1) * control_mask - 1

        # Add batch dimensions
        cond_image_inpaint = cond_image_inpaint.unsqueeze(0)
        control_mask = control_mask.unsqueeze(0)
        text_embeds = text_embeds.unsqueeze(0)
        empty_image = empty_image.unsqueeze(0)
        
        return {
            'cond_image_inpaint': cond_image_inpaint,
            'control_mask': control_mask,
            'prompt': prompt,
            'text_embeds': text_embeds,
            'target_size': (new_h, new_w),
            'controlnet_im': empty_image,
            'embedding_method': 'ids' if self.use_ids else 'character',
        } 