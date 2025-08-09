"""
Utility functions for text processing and coordinate handling in the PosterMaker training pipeline.
"""

import torch
import numpy as np
from typing import List, Tuple, Union


def normalize_coordinates(coords: List[float], width: int, height: int) -> List[float]:
    """
    Normalize coordinates to [0, 1] range based on image dimensions.
    
    Args:
        coords: Coordinates in format [x, y, w, h] or [x1, y1, x2, y2]
        width: Image width
        height: Image height
        
    Returns:
        Normalized coordinates
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


def coords2pos(coords: List[Union[int, float]]) -> List[float]:
    """
    Convert coordinates from [x, y, w, h] format to [x1, y1, x2, y2] format.
    
    Args:
        coords: Coordinates in [x, y, w, h] format
        
    Returns:
        Position in [x1, y1, x2, y2] format
    """
    if len(coords) != 4:
        raise ValueError(f"Expected 4 coordinate values, got {len(coords)}")
    
    x, y, w, h = coords
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    
    return [float(x1), float(y1), float(x2), float(y2)]


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


def mask_image_region(image: torch.Tensor, mask: torch.Tensor, 
                     fill_value: float = 0.0) -> torch.Tensor:
    """
    Mask specific regions of an image.
    
    Args:
        image: Image tensor of shape (C, H, W)
        mask: Binary mask tensor of shape (1, H, W) or (H, W)
        fill_value: Value to fill masked regions
        
    Returns:
        Masked image tensor
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    
    masked_image = image.clone()
    masked_image = masked_image * (1 - mask) + fill_value * mask
    
    return masked_image


def validate_text_annotation(annotation: dict) -> bool:
    """
    Validate text annotation structure.
    
    Args:
        annotation: Annotation dictionary
        
    Returns:
        True if annotation is valid
    """
    required_keys = ['prompt', 'texts']
    
    # Check required keys exist
    for key in required_keys:
        if key not in annotation:
            return False
    
    # Check prompt is not empty
    if not annotation['prompt'] or not isinstance(annotation['prompt'], str):
        return False
    
    # Check texts structure
    if not isinstance(annotation['texts'], list) or len(annotation['texts']) == 0:
        return False
    
    # Validate each text item
    for text_item in annotation['texts']:
        if not isinstance(text_item, dict):
            return False
        
        # Check required fields in text item
        if 'content' not in text_item or 'pos' not in text_item:
            return False
        
        # Validate content
        if not text_item['content'] or not isinstance(text_item['content'], str):
            return False
        
        # Validate position
        pos = text_item['pos']
        if not isinstance(pos, list) or len(pos) != 4:
            return False
        
        try:
            x1, y1, x2, y2 = [int(p) for p in pos]
            if x2 <= x1 or y2 <= y1:
                return False
        except (ValueError, TypeError):
            return False
    
    return True