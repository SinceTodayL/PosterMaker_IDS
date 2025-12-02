import os
import math
from typing import List, Dict, Callable

from PIL import Image
import numpy as np
import cv2

import torch
import Levenshtein

# x1,y1,x2,y2 -> x1,y1,w,h
def pos2coords(pos):
    return (pos[0], pos[1], pos[2]-pos[0], pos[3]-pos[1])


# x1,y1,w,h -> x1,y1,x2,y2
def coords2pos(coords):
    return (coords[0], coords[1], coords[2]+coords[0], coords[3]+coords[1])


def normalize_coordinates(coordinates, original_width, original_height):
    """
    Normalize coordinates to the range [0, 1].
    
    Args:
        coordinates (list): A list of coordinates in the form [x, y, w, h] or [x1, y1, x2, y2].
        original_width (int): The width of the original image.
        original_height (int): The height of the original image.
    
    Returns:
        list: A list of normalized coordinates.
    """
    if len(coordinates) == 4:  # Handle [x, y, w, h] or [x1, y1, x2, y2] format
        x, y, w, h = coordinates
        normalized_coords = [
            x / original_width,
            y / original_height,
            w / original_width,
            h / original_height
        ]
    else:
        raise ValueError("Coordinates must be in the form [x, y, w, h] or [x1, y1, x2, y2].")

    return normalized_coords


def convert_to_rgb(image):
    """
    Convert RGBA or RGB images to RGB format

    Parameters:
    image: numpy.ndarray, Input image (in RGB or RGBA format)

    Returns:
    numpy.ndarray: Image in RGB format
    """
    # If the image is in RGBA format, convert it to RGB
    if image.shape[-1] == 4:
        # Use a white background
        background = np.ones_like(image[..., :3]) * 255
        alpha = image[..., 3:] / 255.0
        image = image[..., :3] * alpha + background * (1 - alpha)
        image = image.astype(np.uint8)
    
    return image[..., :3]  # Ensure the return is in RGB format


def cal_resize_and_padding(img_size, model_input_size):
    ori_h, ori_w = img_size
    target_h, target_w = model_input_size
    
    scale = min(target_h/ori_h, target_w/ori_w)
    new_h, new_w = int(scale * ori_h), int(scale * ori_w)


    return new_h, new_w, scale


def reisize_box_by_scale(box, scale):
    return [int(x * scale) for x in box]


def pad_image_to_shape(image, target_shape, pad_value=0):
    """
    Pad the image to the specified shape.

    Parameters:
    - image: Image to be padded, in numpy array format.
    - target_shape: Target shape (height, width).
    - pad_value: Default value for padding, defaults to 0.

    Returns:
    - Padded image, in numpy array format.
    """
    original_shape = image.shape[:2]
    padding = [
        (0, max(0, target_shape[0] - original_shape[0])),  # Padding in the height direction
        (0, max(0, target_shape[1] - original_shape[1])),  # Padding in the width direction
    ]
    
    if len(image.shape) == 3:  # Check if the image is colored
        padding.append((0, 0))  # Do not pad the channel dimension for colored images
    
    padded_image = np.pad(image, padding, mode='constant', constant_values=pad_value)
    return padded_image
    

def clamp_bbox_to_image(bbox, image_width, image_height):
    """
    Adjusts bounding box coordinates to ensure they do not exceed image boundaries.

    :param bbox: A tuple containing 4 values (x1, y1, x2, y2), representing bounding box coordinates.
    :param image_width: Width of the image.
    :param image_height: Height of the image.
    :return: Adjusted bounding box coordinates (new x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = bbox

    # Ensure bounding box coordinates do not exceed image boundaries
    x1_clamped = max(0, min(x1, image_width))
    y1_clamped = max(0, min(y1, image_height))
    x2_clamped = max(0, min(x2, image_width))
    y2_clamped = max(0, min(y2, image_height))

    return (x1_clamped, y1_clamped, x2_clamped, y2_clamped)


def create_mask_by_text(im_size, texts):
    h, w = im_size
    mask = np.ones((h, w, 1),dtype=np.uint8) * 255
    for text in texts:
        x1, y1, x2, y2 = text['pos']
        mask[y1:y2, x1:x2, :] = 0

    return mask


def get_char_features_by_text(texts, char2feat, char_padding_num, 
                               char_ids_embedder=None, feature_mixer=None,
                               log_stats=False, step=None, print_decomposition=False):
    """
    Get character features for texts, optionally mixing OCR and IDS features
    
    Args:
        texts: List of text dicts with 'content' and 'pos'
        char2feat: OCR feature dictionary
        char_padding_num: Padding length for characters
        char_ids_embedder: Optional CharIDSEmbedder for IDS features
        feature_mixer: Optional LearnableFeatureMixer for mixing OCR and IDS
        log_stats: Whether to log statistics (for debugging)
        step: Current training step (for logging)
        print_decomposition: Whether to print character decomposition info
        
    Returns:
        text_features: List of feature tensors
        token_masks: List of mask tensors
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Print attention status at the beginning if IDS is enabled
    if char_ids_embedder is not None and log_stats:
        attention_status = "ENABLED" if char_ids_embedder.use_attention else "DISABLED"
        num_heads = char_ids_embedder.attention.num_heads if char_ids_embedder.use_attention else 0
        logger.info(f"[IDS Config] Attention mechanism: {attention_status}" + (f" (num_heads={num_heads})" if char_ids_embedder.use_attention else ""))
    
    text_features = []
    token_masks = []
    
    # Statistics for logging
    stats = {
        'total_chars': 0,
        'ids_processed': 0,
        'ids_failed': 0,
        'alpha_values': [],
        'ocr_norms': [],
        'ids_norms': [],
        'mixed_norms': [],
        'sample_chars': []
    }

    for text in texts:
        content = text['content']
        # Get default feature
        default_feature = char2feat[' '][None, ...]

        # Pre-allocate space and fill with default features
        char_features = torch.empty((len(content), *default_feature.shape[1:]), dtype=default_feature.dtype)
        default_val = default_feature.squeeze(0)
        
        # Process each character
        for i, c in enumerate(content):
            stats['total_chars'] += 1
            
            # Get OCR feature (always in no_grad for OCR features)
            with torch.no_grad():
                if c in char2feat:
                    ocr_feat = char2feat[c]
                else:
                    ocr_feat = default_val
                ocr_norm = torch.norm(ocr_feat).item() if isinstance(ocr_feat, torch.Tensor) else torch.norm(torch.tensor(ocr_feat)).item()
                stats['ocr_norms'].append(ocr_norm)
            
            # If IDS mixing is enabled, get IDS feature and mix (needs gradients)
            if char_ids_embedder is not None and feature_mixer is not None:
                try:
                    # Get IDS feature for this character (with gradients)
                    # Print decomposition for first few characters or when explicitly requested
                    should_print = print_decomposition and (stats['total_chars'] <= 5 or stats['total_chars'] % 20 == 0)
                    ids_feat = char_ids_embedder.encode_char(c, print_decomposition=should_print)  # (64,)
                    ids_norm = torch.norm(ids_feat).item()
                    stats['ids_norms'].append(ids_norm)
                    stats['ids_processed'] += 1
                    
                    # Get current alpha value
                    alpha = feature_mixer.get_alpha()
                    stats['alpha_values'].append(alpha)
                    
                    # Ensure same device and dtype for OCR feature
                    if isinstance(ocr_feat, torch.Tensor):
                        ocr_feat_tensor = ocr_feat.to(device=ids_feat.device, dtype=ids_feat.dtype)
                    else:
                        ocr_feat_tensor = torch.tensor(ocr_feat, device=ids_feat.device, dtype=ids_feat.dtype)
                    
                    # Mix features using learnable mixer (with gradients)
                    mixed_feat = feature_mixer(ocr_feat_tensor, ids_feat)  # (64,)
                    mixed_norm = torch.norm(mixed_feat).item()
                    stats['mixed_norms'].append(mixed_norm)
                    
                    # Sample a few characters for detailed logging
                    if log_stats and len(stats['sample_chars']) < 3 and stats['total_chars'] % 10 == 0:
                        stats['sample_chars'].append({
                            'char': c,
                            'ocr_norm': ocr_norm,
                            'ids_norm': ids_norm,
                            'mixed_norm': mixed_norm,
                            'alpha': alpha
                        })
                    
                    char_features[i] = mixed_feat
                except Exception as e:
                    stats['ids_failed'] += 1
                    if log_stats:
                        logger.warning(f"IDS encoding failed for char '{c}': {e}")
                    # Fallback to OCR only
                    if not isinstance(ocr_feat, torch.Tensor):
                        ocr_feat = torch.tensor(ocr_feat)
                    char_features[i] = ocr_feat
            else:
                # Pure OCR feature (backward compatible)
                # Ensure it's a tensor
                if not isinstance(ocr_feat, torch.Tensor):
                    ocr_feat = torch.tensor(ocr_feat)
                char_features[i] = ocr_feat

        # Get shape information
        N = char_features.shape[0]

        # Use zeros to create a padding tensor for concatenation
        padding_tensor = torch.zeros((char_padding_num - N, *char_features.shape[1:]), dtype=char_features.dtype)

        # Concatenation
        char_features = torch.cat([char_features, padding_tensor], dim=0)

        assert char_features.shape[0] == char_padding_num, "len(char_features) == padding_to_len"

        text_features.append(char_features)

        # char token mask
        char_token_mask = torch.zeros(char_padding_num)

        char_token_mask[:N] = 1

        token_masks.append(char_token_mask)
    
    # Log statistics if requested
    if log_stats and char_ids_embedder is not None and feature_mixer is not None:
        if stats['ids_processed'] > 0:
            avg_alpha = sum(stats['alpha_values']) / len(stats['alpha_values'])
            avg_ocr_norm = sum(stats['ocr_norms']) / len(stats['ocr_norms']) if stats['ocr_norms'] else 0
            avg_ids_norm = sum(stats['ids_norms']) / len(stats['ids_norms']) if stats['ids_norms'] else 0
            avg_mixed_norm = sum(stats['mixed_norms']) / len(stats['mixed_norms']) if stats['mixed_norms'] else 0
            
            log_msg = f"[Step {step}] IDS Stats: "
            log_msg += f"chars={stats['total_chars']}, ids_ok={stats['ids_processed']}, ids_failed={stats['ids_failed']}, "
            log_msg += f"alpha={avg_alpha:.4f}, ocr_norm={avg_ocr_norm:.4f}, ids_norm={avg_ids_norm:.4f}, mixed_norm={avg_mixed_norm:.4f}"
            logger.info(log_msg)
            
            if stats['sample_chars']:
                for sample in stats['sample_chars']:
                    logger.info(f"  Sample char '{sample['char']}': alpha={sample['alpha']:.4f}, "
                              f"ocr_norm={sample['ocr_norm']:.4f}, ids_norm={sample['ids_norm']:.4f}, "
                              f"mixed_norm={sample['mixed_norm']:.4f}")
            
    return text_features, token_masks


# define cosine position encoding function
def get_positional_encoding(length, channels):
    position = np.arange(length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, channels, 2) * -(np.log(10000.0) / channels))
    pe = np.zeros((length, channels))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(pe)


def save_image(image, im_path):
    if isinstance(image, Image.Image):
        image = np.array(image) 
    elif isinstance(image, np.ndarray):
        image = image.astype(np.uint8)
    else:
        raise ValueError("image must be PIL.Image.Image or numpy.ndarray")

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(im_path, image)


def post_process(image, target_size):
    """
    对模型输出结果进行后处理
    
    Args:
        image: 模型的原始输出结果，是PIL.Image格式
        target_size: 目标尺寸，包含(height, width)的元组或列表
        
    Returns:
        PIL.Image: 裁剪后的图像
    """
    im_h, im_w = target_size
    
    # 裁剪图像到目标尺寸
    crop_rel = image.crop((0, 0, im_w, im_h))
    
    return crop_rel


def read_im(path, root=''):
    img_dir = os.path.join(root, path)
    try:
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #bgr -> rgb
    except:
        print(f"image read error: {img_dir}")
        img = None

    return img


def sort_texts_by_pos(texts):
    """
    对texts的列表进行排序，依据是pos中的x1, y1, x2, y2依次进行排序
    """
    try:
        # 使用sorted函数进行排序，key参数用于指定排序依据
        sorted_texts = sorted(texts, key=lambda x: (x['pos'][0], x['pos'][1], x['pos'][2], x['pos'][3]))
        return sorted_texts
    except Exception as e:
        print(e)
        return None


def copy_text_to_bg(poster_im, gt_im, texts):
    for text in texts:
        x1, y1, x2, y2 = text['pos']
        gt_im[y1:y2, x1:x2, :] = poster_im[y1:y2, x1:x2, :].copy()
    return gt_im

def mask_image_by_texts(im, texts, mask_value=0):
    im_mask = im.copy()
    for text in texts:
        x1, y1, x2, y2 = text['pos']
        if im_mask.ndim == 2:  # 灰度图像
            im_mask[y1:y2, x1:x2] = mask_value
        else:  # RGB或其他有多个通道的图像
            im_mask[y1:y2, x1:x2, ...] = mask_value

    return im_mask

def mask_image_by_logos(im, logos, mask_value=0):
    im_mask = im.copy()
    for logo in logos:
        x1, y1, x2, y2 = logo
        if im_mask.ndim == 2:  # 灰度图像
            im_mask[y1:y2, x1:x2] = mask_value
        else:  # RGB或其他有多个通道的图像
            im_mask[y1:y2, x1:x2, ...] = mask_value

    return im_mask

def check_and_create_directory(directory_path):
    # 如果文件夹存在
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def full_to_half_width(s):
    n = []
    for char in s:
        code = ord(char)
        # 处理全角字符（除空格）编码范围
        if 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        # 处理全角空格（占两个字符位置）
        elif code == 0x3000:
            code = 0x0020
        # 处理全角双引号
        elif code == 0x201C:  # “
            code = 0x0022
        elif code == 0x201D:  # ”
            code = 0x0022
        n.append(chr(code))
    return ''.join(n)


def get_ld(ls1, ls2):
    edit_dist = Levenshtein.distance(ls1, ls2)
    return 1 - edit_dist/(max(len(ls1), len(ls2)) + 1e-5)


def pre_process(img_list, shape):
    numpy_list = []
    img_num = len(img_list)
    assert img_num > 0
    for idx in range(0, img_num):
        # rotate
        img = img_list[idx]
        h, w = img.shape[1:]
        if h > w * 1.2:
            img = torch.transpose(img, 1, 2).flip(dims=[1])
            img_list[idx] = img
            h, w = img.shape[1:]
        # resize
        imgC, imgH, imgW = (int(i) for i in shape.strip().split(','))
        assert imgC == img.shape[0]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(imgH, resized_w),
            mode='bilinear',
            align_corners=True,
        )
        # padding
        padding_im = torch.zeros((imgC, imgH, imgW), dtype=torch.float32)
        padding_im[:, :, 0:resized_w] = resized_image[0]
        numpy_list += [padding_im.permute(1, 2, 0).cpu().numpy()]  # HWC ,numpy
    return numpy_list


def check_layout(pos: list, content: str, poslist: list, url: str) -> bool:
    if not url and  not content:
        return False
    
    h = abs(pos[1]-pos[3])
    w = abs(pos[0]-pos[2])
    len_text = len(content)
    num_text = len(poslist)
    
    return 0 <= pos[0] and 0 <= pos[1] and w >= h and h >= 20 and len_text <= 16


def filter_samples(samples: List[Dict],
                   check_layout: Callable[[list, str, list, str], bool]
                  ) -> List[Dict]:
    """
    Args:
        samples: List[dict]，每个dict格式参考你的描述
        check_layout: 一个函数，签名为 (pos, content, poslist, url) -> bool

    Returns:
        new_samples: 过滤后的samples（list of dict），只保留texts非空项
    """
    filtered_samples = []
    for sample in samples:
        texts_keep = []
        texts_del = []
        texts = sample.get('texts', [])
        poslist = [t['pos'] for t in texts]
        url = sample.get('url', "")
        for text in texts:
            pos = text['pos']
            content = text['content']
            if check_layout(pos=pos, content=content, poslist=poslist, url=url):
                texts_keep.append(text)
            else:
                texts_del.append(text)
        if texts_keep:  # 只保留有剩余texts的sample
            sample_new = dict(sample)  # 浅拷贝以免影响原数据
            sample_new['texts'] = texts_keep
            sample_new['texts_out'] = texts_del if texts_del else None
            filtered_samples.append(sample_new)
            
    return filtered_samples
