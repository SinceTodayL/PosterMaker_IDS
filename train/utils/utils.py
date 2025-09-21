"""
utility functions
"""

import torch
import numpy as np
from typing import List, Tuple, Union
from PIL import Image, ImageOps


def normalize_coordinates(coords: List[float], width: int, height: int) -> List[float]:
    """
    Normalize coordinates to [0, 1] range based on image dimensions.
    
    Args:
        coords: Coordinates in format [x, y, w, h]
        width: Image width
        height: Image height
        
    Returns:
        Normalized coordinates [x/width, y/height, w/width, h/height]
    """
    if len(coords) == 4:
        x, y, w, h = coords
        return [x / width, y / height, w / width, h / height]
    else:
        raise ValueError(f"Expected 4 coordinates, got {len(coords)}")


def pos2coords(pos: List[Union[int, float]]) -> List[float]:
    """
    Convert position from [x1, y1, x2, y2] format to [x, y, w, h] format.
    
    Args:
        pos: Position coordinates in [x1, y1, x2, y2] format
        
    Returns:
        Coordinates in [x, y, w, h] format
    """
    if len(pos) != 4:
        raise ValueError(f"Expected 4 position values, got {len(pos)}")
    
    x1, y1, x2, y2 = pos
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1
    
    return [float(x), float(y), float(w), float(h)]


def get_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    Generate sinusoidal positional encodings.
    
    Args:
        max_len: Maximum sequence length
        d_model: Embedding dimension
        
    Returns:
        Positional encoding tensor of shape (max_len, d_model)
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        (-np.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe


def create_text_mask(text_positions: List[List[int]], image_size: Tuple[int, int]) -> torch.Tensor:
    """
    Create a mask for text regions in the image.
    
    Args:
        text_positions: List of text box positions in [x1, y1, x2, y2] format
        image_size: (height, width) of the image
        
    Returns:
        Binary mask tensor of shape (1, height, width)
    """
    height, width = image_size
    mask = torch.zeros(1, height, width)
    
    for pos in text_positions:
        x1, y1, x2, y2 = pos
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        if x2 > x1 and y2 > y1:
            mask[:, y1:y2, x1:x2] = 1.0
    
    return mask


def resize_keep_ratio(image: Image.Image, target_size: Tuple[int, int]) -> Tuple[Image.Image, float, int, int]:
    """Resize image keeping aspect ratio and pad to target size (PosterMaker style)"""
    target_w, target_h = target_size
    orig_w, orig_h = image.size
    
    scale = min(target_h / orig_h, target_w / orig_w)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    
    # Resize image (use BILINEAR for images, NEAREST for masks to preserve mask values)
    if image.mode == 'L':
        resized = image.resize((new_w, new_h), Image.NEAREST)  # Preserve binary mask values
    else:
        resized = image.resize((new_w, new_h), Image.BILINEAR)  # Smooth interpolation for images
    
    # Create background with white fill (PosterMaker style)
    if image.mode == 'RGB':
        padded = Image.new('RGB', (target_w, target_h), (255, 255, 255))
    else:  # L mode for masks  
        padded = Image.new(image.mode, (target_w, target_h), 255)
    
    # Paste at top-left corner (0, 0) - PosterMaker style
    padded.paste(resized, (0, 0))
    
    # Return actual padding info (top-left alignment means right/bottom padding)
    pad_right = target_w - new_w
    pad_bottom = target_h - new_h
    return padded, scale, pad_right, pad_bottom