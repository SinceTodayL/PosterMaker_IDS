import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from utils.utils import create_text_mask, resize_keep_ratio

class PosterDataset(Dataset):
    def __init__(self, dataset_dir, split='train', target_size=(1024, 1024)):
        self.dataset_dir = os.path.join(dataset_dir, split)
        self.target_size = target_size
        self.samples = []
        
        for item in os.listdir(self.dataset_dir):
            item_path = os.path.join(self.dataset_dir, item)
            if os.path.isdir(item_path):
                annotation_path = os.path.join(item_path, 'annotation.json')
                mask_path = os.path.join(item_path, 'subject_mask.png')
                
                # Support both .jpg and .png image formats
                image_path_jpg = os.path.join(item_path, 'image.jpg')
                image_path_png = os.path.join(item_path, 'image.png')
                
                if os.path.exists(image_path_jpg):
                    image_path = image_path_jpg
                elif os.path.exists(image_path_png):
                    image_path = image_path_png
                else:
                    continue  # Skip this sample if no image file found
                
                if all(os.path.exists(p) for p in [annotation_path, mask_path]):
                    self.samples.append({
                        'annotation_path': annotation_path,
                        'image_path': image_path,
                        'mask_path': mask_path
                    })
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        with open(sample['annotation_path'], 'r', encoding='utf-8') as f:
            annotation = json.load(f)
        
        image = Image.open(sample['image_path']).convert('RGB')
        mask = Image.open(sample['mask_path']).convert('L')
        
        # Resize with aspect ratio preservation + padding
        padded_img, scale, pad_left, pad_top = resize_keep_ratio(image, self.target_size)
        pixel_values = self.transform(padded_img)
        
        padded_mask, _, _, _ = resize_keep_ratio(mask, self.target_size)
        subject_mask = transforms.ToTensor()(padded_mask)
        
        # Transform text positions (no padding offset needed for top-left alignment)
        text_positions = []
        for text_info in annotation['texts']:
            x1, y1, x2, y2 = text_info['pos']
            x1_new = int(x1 * scale)
            y1_new = int(y1 * scale)
            x2_new = int(x2 * scale)
            y2_new = int(y2 * scale)
            text_positions.append([x1_new, y1_new, x2_new, y2_new])
        
        text_mask = create_text_mask(text_positions, (self.target_size[1], self.target_size[0]))
        
        # Update coordinates for ids_text_embedder
        transformed_texts = []
        for i, text_info in enumerate(annotation['texts']):
            transformed_texts.append({
                'content': text_info['content'],
                'pos': text_positions[i]
            })
        
        # === PosterMaker双ControlNet架构修复 ===
        # SceneGenNet: 接收被subject_mask处理的图像，学习场景补全/inpainting
        # subject_mask: 1=保留主体, 0=需要生成背景
        # 正确公式：scene_conditioning = pixel_values * subject_mask + (-1.0) * (1.0 - subject_mask)
        scene_conditioning_pixel_values = pixel_values * subject_mask + (-1.0) * (1.0 - subject_mask)
        
        # TextRenderNet: 接收完全空白的图像，学习纯文本渲染
        text_render_conditioning_pixel_values = torch.full_like(pixel_values, -1.0)
        
        return {
            'pixel_values': pixel_values,
            'scene_conditioning_pixel_values': scene_conditioning_pixel_values,
            'text_render_conditioning_pixel_values': text_render_conditioning_pixel_values,
            'subject_mask': subject_mask,  
            'text_mask': text_mask,        
            'conditioning_mask': text_mask,  
            'prompt': annotation['prompt'],
            'all_texts_info': transformed_texts,
            'text_content': ' '.join([t['content'] for t in annotation['texts']])
        }

def create_dataloaders(config):
    dataset_dir = config['dataset_dir']
    batch_size = config['stage1']['batch_size']
    target_size = tuple(config.get('image_size', [1024, 1024]))
    
    train_dataset = PosterDataset(dataset_dir, 'train', target_size)
    val_dataset = PosterDataset(dataset_dir, 'val', target_size)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader
