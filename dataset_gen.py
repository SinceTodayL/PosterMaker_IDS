#!/usr/bin/env python3
"""
Generate training dataset using Qwen-VL API.
Cleans existing data (removes samples without captions) and generates new captions.
Can run in background with nohup.
"""

import os
import json
import argparse
import base64
from pathlib import Path
from tqdm import tqdm
import requests
from typing import List, Dict, Optional

# Paths
POSTER_DIR = Path('../dataset/poster')
POSTER_TXT = POSTER_DIR / 'poster.txt'
TRAINING_DIR = Path('../dataset/training_poster')
TRAIN_DATA_JSON = TRAINING_DIR / 'train_data.json'
GT_IMAGES_DIR = TRAINING_DIR / 'images' / 'gt'
CONFIG_FILE = Path('config.json')
QWEN_VL_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


def load_config():
    """Load API key from config.json"""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    api_key = config.get('qwen_vl_apikey') or config.get('api_key')
    if not api_key:
        raise ValueError("API key not found in config.json")
    return api_key


def clean_dataset():
    """Remove samples without captions from train_data.json"""
    if not TRAIN_DATA_JSON.exists():
        
        print("No existing train_data.json found, will create new one.")
        return
    
    with open(TRAIN_DATA_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    cleaned_data = [s for s in data if 'caption' in s and s.get('caption')]
    
    if len(cleaned_data) < original_count:
        backup_path = TRAIN_DATA_JSON.with_suffix('.json.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        with open(TRAIN_DATA_JSON, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        print(f"Cleaned dataset: {original_count} -> {len(cleaned_data)} samples (removed {original_count - len(cleaned_data)} without captions)")
        print(f"Backup saved to: {backup_path}")
    else:
        print(f"Dataset already clean: {len(cleaned_data)} samples with captions")


def image_to_base64(image_path: Path) -> str:
    """Convert image to base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def call_qwen_vl_api(image_path: Path, api_key: str) -> Optional[str]:
    """Call Qwen-VL API to generate caption"""
    try:
        image_base64 = image_to_base64(image_path)
        prompt = """Describe this product poster image focusing only on visual elements: 
the main product/subject, background, decorations, colors, and overall scene composition. 
DO NOT mention any text, words, or characters visible in the image. 
Use 'The Subject' to refer to the main product."""
        
        payload = {
            "model": "qwen-vl-max",
            "input": {
                "messages": [{
                    "role": "user",
                    "content": [
                        {"image": f"data:image/jpeg;base64,{image_base64}"},
                        {"text": prompt}
                    ]
                }]
            },
            "parameters": {"max_tokens": 150, "temperature": 0.7}  # Allow enough tokens for ~70 words (tokens > words)
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(QWEN_VL_API_URL, json=payload, headers=headers, timeout=30)
        print(f"    [API Response] Status: {response.status_code}", flush=True)
        response.raise_for_status()
        result = response.json()
        
        if 'output' in result and 'choices' in result['output']:
            if len(result['output']['choices']) > 0:
                content = result['output']['choices'][0]['message']['content']
                # Handle both string and list formats
                if isinstance(content, str):
                    caption = content.strip()
                elif isinstance(content, list):
                    # Extract text from list of dicts
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            text_parts.append(item['text'])
                        elif isinstance(item, str):
                            text_parts.append(item)
                    caption = ' '.join(text_parts).strip()
                else:
                    caption = str(content).strip()
                
                if caption:
                    # Limit to 70 words maximum
                    words = caption.split()
                    if len(words) > 70:
                        caption = ' '.join(words[:70])
                        print(f"    [API Warning] Caption truncated from {len(words)} to 70 words", flush=True)
                    print(f"    [API Success] Caption: {len(words)} words, {len(caption)} chars", flush=True)
                    return caption
        print(f"    [API Warning] Unexpected response format", flush=True)
        return None
    except requests.exceptions.RequestException as e:
        print(f"    [API Error] Request failed for {image_path.name}: {e}", flush=True)
        return None
    except Exception as e:
        print(f"    [API Error] Exception for {image_path.name}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None


def find_image_file(image_name: str, poster_dir: Path, subdirs: List[str]) -> Optional[Path]:
    """Find image file in specified subdirectories"""
    for subdir in subdirs:
        image_path = poster_dir / subdir / image_name
        if image_path.exists():
            return image_path
    return None


def validate_caption(caption: str, texts: List[Dict]) -> bool:
    """Validate caption doesn't contain text content"""
    if not caption:
        return False
    caption_lower = caption.lower()
    for text in texts:
        content = text['content'].lower()
        if len(content) > 3 and content in caption_lower:
            return False
    return True


def discover_available_subdirs(poster_dir: Path) -> List[str]:
    """Discover available poster subdirectories"""
    subdirs = []
    if poster_dir.exists():
        for item in poster_dir.iterdir():
            if item.is_dir() and item.name.startswith('poster_'):
                subdirs.append(item.name)
    return sorted(subdirs)


def generate_dataset(subdirs: List[str] = None, skip_existing: bool = True):
    """Generate training dataset"""
    # Clean existing data first
    clean_dataset()
    
    if subdirs is None:
        subdirs = ['poster_0']
    
    # Validate subdirs
    available_subdirs = discover_available_subdirs(POSTER_DIR)
    invalid_subdirs = [s for s in subdirs if s not in available_subdirs]
    if invalid_subdirs:
        raise ValueError(f"Invalid subdirectories: {invalid_subdirs}. Available: {available_subdirs}")
    
    print(f"Using subdirectories: {subdirs}")
    
    # Create directories
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    GT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load API key
    api_key = load_config()
    
    # Load existing data
    existing_samples = {}
    if skip_existing and TRAIN_DATA_JSON.exists():
        with open(TRAIN_DATA_JSON, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            existing_samples = {item['url']: item for item in existing_data}
        print(f"Loaded {len(existing_samples)} existing samples")
    
    # Process poster.txt
    samples = []
    missing_images = []
    failed_captions = []
    processed_urls = set()
    save_interval = 10  # Save every N samples
    
    # Pre-filter: only process images that exist in specified subdirs
    print("Pre-filtering images in specified subdirectories...")
    available_images = set()
    for subdir in subdirs:
        subdir_path = POSTER_DIR / subdir
        if subdir_path.exists():
            for img_file in subdir_path.glob('*.jpg'):
                available_images.add(img_file.name)
            for img_file in subdir_path.glob('*.png'):
                available_images.add(img_file.name)
    print(f"Found {len(available_images)} images in {subdirs}")
    
    with open(POSTER_TXT, 'r', encoding='utf-8') as f:
        lines = list(f)
    
    # Filter lines to only those with images in specified subdirs, deduplicate by image name
    filtered_lines = []
    seen_images = set()
    duplicate_count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            image_name = data.get('name', '')
            if image_name in available_images:
                if image_name not in seen_images:
                    filtered_lines.append(line)
                    seen_images.add(image_name)
                else:
                    duplicate_count += 1
        except:
            continue
    
    print(f"Filtered to {len(filtered_lines)} unique images matching {subdirs}")
    if duplicate_count > 0:
        print(f"  (Skipped {duplicate_count} duplicate entries in poster.txt)")
    print(f"Processing {len(filtered_lines)} samples (saving every {save_interval} samples)...")
    
    def save_samples():
        """Save current samples to file"""
        try:
            # Load existing data
            all_samples = {}
            if TRAIN_DATA_JSON.exists():
                try:
                    with open(TRAIN_DATA_JSON, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                        all_samples = {item['url']: item for item in existing}
                except Exception as e:
                    print(f"  [Warning] Error loading existing data: {e}", flush=True)
            
            # Merge new samples
            for sample in samples:
                all_samples[sample['url']] = sample
            
            # Save
            with open(TRAIN_DATA_JSON, 'w', encoding='utf-8') as f:
                json.dump(list(all_samples.values()), f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"  [Error saving] {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False
    
    for line_num, line in enumerate(tqdm(filtered_lines, desc="Processing"), 1):
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
            image_name = data.get('name', '')
            if not image_name or image_name in processed_urls:
                continue
            
            # Skip if exists with caption
            if skip_existing and image_name in existing_samples:
                if 'caption' in existing_samples[image_name]:
                    samples.append(existing_samples[image_name])
                    processed_urls.add(image_name)
                    continue
            
            # Find image
            image_path = find_image_file(image_name, POSTER_DIR, subdirs)
            if image_path is None:
                missing_images.append((line_num, image_name))
                continue
            
            # Generate caption
            print(f"  [API Call] Processing {image_name}...", flush=True)
            caption = call_qwen_vl_api(image_path, api_key)
            if not caption:
                print(f"  [API Failed] No caption generated for {image_name}", flush=True)
                failed_captions.append((line_num, image_name))
                continue
            print(f"  [API Success] Caption generated for {image_name}", flush=True)
            
            # Validate caption
            texts = data.get('texts', [])
            if not validate_caption(caption, texts):
                caption = call_qwen_vl_api(image_path, api_key)
                if not caption or not validate_caption(caption, texts):
                    failed_captions.append((line_num, image_name))
                    continue
            
            # Create sample
            sample = {'url': image_name, 'caption': caption, 'texts': texts}
            if 'logo' in data:
                sample['logo'] = data['logo']
            if 'texts_out' in data:
                sample['texts_out'] = data['texts_out']
            
            samples.append(sample)
            processed_urls.add(image_name)
            
            # Copy image
            dest_path = GT_IMAGES_DIR / image_name
            if not dest_path.exists():
                import shutil
                shutil.copy2(image_path, dest_path)
            
            # Save periodically (every N samples)
            if len(samples) % save_interval == 0:
                try:
                    save_samples()
                    print(f"  [Saved] {len(samples)} new samples processed", flush=True)
                except Exception as e:
                    print(f"  [Error saving] {e}", flush=True)
        
        except Exception as e:
            print(f"Error at line {line_num}: {e}")
            continue
    
    # Final save
    if samples:
        save_samples()
        print(f"\nFinal save: {len(samples)} new samples saved")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total lines in poster.txt: {len(lines)}")
    print(f"Lines matching specified subdirs: {len(filtered_lines)}")
    print(f"Unique URLs processed: {len(processed_urls)}")
    print(f"Valid samples with captions: {len(samples)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Failed captions: {len(failed_captions)}")
    print(f"Training data saved to: {TRAIN_DATA_JSON}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Generate training dataset using Qwen-VL API')
    parser.add_argument('--subdirs', type=str, nargs='+', default=None,
                       help='Subdirectories (e.g., poster_0 poster_4 poster_6). Default: poster_0')
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='Regenerate all captions')
    parser.add_argument('--list-subdirs', action='store_true',
                       help='List available subdirectories and exit')
    
    args = parser.parse_args()
    
    if args.list_subdirs:
        available = discover_available_subdirs(POSTER_DIR)
        print("Available subdirectories:")
        for subdir in available:
            print(f"  - {subdir}")
        return
    
    generate_dataset(
        subdirs=args.subdirs or ['poster_0'],
        skip_existing=not args.no_skip_existing
    )


if __name__ == '__main__':
    main()


