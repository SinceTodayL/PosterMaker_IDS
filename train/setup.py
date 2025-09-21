import torch
import yaml
import os
import random
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import DDPMScheduler

from utils.ids_tokenizer import IDSTokenizer
from models.ids_text_embedder import IDSTextEmbedder
from models.adapter_models import LinearAdapterWithLayerNorm
from models.controlnet_sd3 import SD3ControlNetModel
from dataset import create_dataloaders

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required keys
    required_keys = ['poster_maker_dir', 'dataset_dir', 'output_dir', 'stage1', 'seed']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate paths
    if not os.path.exists(config['poster_maker_dir']):
        raise FileNotFoundError(f"PosterMaker directory not found: {config['poster_maker_dir']}")
    
    if not os.path.exists(config['dataset_dir']):
        raise FileNotFoundError(f"Dataset directory not found: {config['dataset_dir']}")
    
    # Create output directory if not exists
    os.makedirs(config['output_dir'], exist_ok=True)
    
    return config

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # Set CUDA memory allocation strategy to reduce fragmentation
        torch.cuda.empty_cache()
        # Enable memory pool for better memory management
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    return device

def create_models(config, device):
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    from diffusers import AutoencoderKL, SD3Transformer2DModel
    
    poster_maker_dir = config['poster_maker_dir']
    sd3_path = os.path.join(poster_maker_dir, config['sd3_model_path'])
    
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(sd3_path, subfolder="vae", torch_dtype=torch.float16)
    
    print("Loading main transformer...")
    transformer = SD3Transformer2DModel.from_pretrained(sd3_path, subfolder="transformer", torch_dtype=torch.float16)
    
    print("Loading text encoders...")
    # Load text encoders in FP32 to avoid LayerNorm issues on CPU
    text_encoder = CLIPTextModelWithProjection.from_pretrained(sd3_path, subfolder="text_encoder", torch_dtype=torch.float32)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(sd3_path, subfolder="text_encoder_2", torch_dtype=torch.float32)
    text_encoder_3 = T5EncoderModel.from_pretrained(sd3_path, subfolder="text_encoder_3", torch_dtype=torch.float32)
    
    tokenizer = CLIPTokenizer.from_pretrained(sd3_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(sd3_path, subfolder="tokenizer_2") 
    tokenizer_3 = T5TokenizerFast.from_pretrained(sd3_path, subfolder="tokenizer_3")
    
    # Clear cache before loading ControlNets
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print("Creating ControlNet models...")
    scene_gen_net = SD3ControlNetModel.from_transformer(
        transformer, additional_in_channel=1, load_weights_from_transformer=True
    )
    text_render_net = SD3ControlNetModel.from_transformer(
        transformer, additional_in_channel=0, load_weights_from_transformer=True
    )
    
    # Clear cache after creating ControlNets
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Load custom weights if available
    scene_gen_path = os.path.join(poster_maker_dir, config['scene_gen_net_path'])
    if os.path.exists(scene_gen_path):
        print(f"Loading SceneGenNet weights from {scene_gen_path}")
        scene_checkpoint = torch.load(scene_gen_path, map_location='cpu')
        if 'state_dict' in scene_checkpoint:
            scene_gen_net.load_state_dict(scene_checkpoint['state_dict'], strict=False)
        else:
            scene_gen_net.load_state_dict(scene_checkpoint, strict=False)
        del scene_checkpoint
    
    text_render_path = os.path.join(poster_maker_dir, config['original_textrender_net_path'])
    if os.path.exists(text_render_path):
        print(f"Loading TextRenderNet weights from {text_render_path}")
        text_checkpoint = torch.load(text_render_path, map_location='cpu')
        if 'state_dict' in text_checkpoint:
            text_render_net.load_state_dict(text_checkpoint['state_dict'], strict=False)
        else:
            text_render_net.load_state_dict(text_checkpoint, strict=False)
        del text_checkpoint
    
    # Create IDS tokenizer and models
    print("Creating IDS models...")
    ids_tokenizer = IDSTokenizer(config['ids_database_path'], config.get('ids_vocab_path'))
    
    ids_text_embedder = IDSTextEmbedder(
        ids_database_path=config['ids_database_path'],
        vocab_file=config.get('ids_vocab_path'),
        ids_embed_dim=config['stage1']['embedding_dim'],
        max_seq_length=config['stage1']['max_seq_length'],
        char2feat_path=config['char2feat_path'],
        input_size=tuple(config.get('image_size', [1024, 1024]))
    )
    
    # The adapter transforms IDS embedder token features (128D) to SD3 joint attention dimension
    # IDS text embedder outputs 128D features (64 + 32 + 32)
    adapter = LinearAdapterWithLayerNorm(
        hidden_dim=128,  # FIXED: IDS text embedder output dimension
        projection_dim=transformer.config.joint_attention_dim
    )
    
    # Apply the successful model loading pattern from working version
    models = {
        'vae': vae.to(device=device, dtype=torch.float16),  # Keep VAE on GPU
        'transformer': transformer.to(device=device, dtype=torch.float16), 
        'text_render_net': text_render_net.to(device=device, dtype=torch.float32),  # FP32 for trainable model in AMP
        'scene_gen_net': scene_gen_net.to(device=device, dtype=torch.float16),
        # Keep trainable modules in FP32 on GPU for training stability
        'ids_text_embedder': ids_text_embedder.to(device=device, dtype=torch.float32),
        'adapter': adapter.to(device=device, dtype=torch.float32),
        # Offload text encoders to CPU to reduce GPU memory footprint
        'text_encoder': text_encoder.to("cpu"),
        'text_encoder_2': text_encoder_2.to("cpu"),
        'text_encoder_3': text_encoder_3.to("cpu"),
    }
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(device) / (1024**3)
        print(f"GPU Memory after model loading: {allocated_memory:.2f} GB allocated, {reserved_memory:.2f} GB reserved")
        print("Model device allocation:")
        for name, model in models.items():
            model_device = next(model.parameters()).device
            print(f"  {name}: {model_device}")
        print(f"Memory optimized: Text encoders moved to CPU (saved ~7.8GB)")
    
    # Initialize trainable model weights to prevent gradient explosion
    def init_weights(m):
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.02)  # Very small gain
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, (torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm2d)):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
    
    # Apply careful initialization to trainable models only
    models['adapter'].apply(init_weights)
    
    # Freeze non-trainable models (except text_render_net which we need for gradient flow)
    for name, model in models.items():
        if name not in ['ids_text_embedder', 'adapter', 'text_render_net']:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
    
    # SMART STRATEGY: Only train the LAST FEW LAYERS of text_render_net
    # This allows gradient flow but drastically reduces trainable parameters
    text_render_net = models['text_render_net']
    
    # First, freeze ALL parameters
    for param in text_render_net.parameters():
        param.requires_grad = False
    
    # Then, selectively enable training for output layers only
    trainable_layer_count = 0
    max_trainable_layers = 3  # Only train the last 3 layers
    
    # Go through layers in reverse order and enable training for the last few
    all_modules = list(text_render_net.named_modules())
    for name, module in reversed(all_modules):
        if trainable_layer_count >= max_trainable_layers:
            break
            
        # Enable training for output/final layers (typically contain 'out', 'final', or are linear/conv layers)
        if any(keyword in name.lower() for keyword in ['out', 'final', 'head']) or \
           isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)) and len(list(module.parameters())) > 0:
            for param in module.parameters():
                param.requires_grad = True
            trainable_layer_count += 1
            print(f"Enabled training for text_render_net layer: {name}")
    
    # Count trainable parameters in text_render_net
    trainable_params = sum(p.numel() for p in text_render_net.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in text_render_net.parameters())
    print(f"TextRenderNet: {trainable_params:,} trainable / {total_params:,} total ({trainable_params/total_params*100:.2f}%)")
    
    tokenizers = {
        'tokenizer': tokenizer,
        'tokenizer_2': tokenizer_2,
        'tokenizer_3': tokenizer_3
    }
    
    return models, tokenizers

def create_optimizer_scheduler(models, config):
    trainable_params = []
    for name in ['ids_text_embedder', 'adapter', 'text_render_net']:
        # Only add parameters that actually require gradients
        trainable_params.extend([p for p in models[name].parameters() if p.requires_grad])
    
    stage1_config = config['stage1']
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=stage1_config['learning_rate'],
        weight_decay=stage1_config['weight_decay']
    )
    
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=stage1_config['num_train_epochs'],
        eta_min=stage1_config['learning_rate'] * 0.01
    )
    
    return optimizer, lr_scheduler

def setup_all(config_path: str):
    try:
        print(f"Loading configuration from {config_path}...")
        config = load_config(config_path)
        
        print(f"Setting random seed to {config['seed']}...")
        set_seed(config['seed'])
        
        print("Setting up device...")
        device = setup_device()
        print(f"Using device: {device}")
        
        print("Loading models...")
        models, tokenizers = create_models(config, device)
        
        print("Creating dataloaders...")
        train_dataloader, val_dataloader = create_dataloaders(config)
        print(f"Training samples: {len(train_dataloader.dataset)}")
        print(f"Validation samples: {len(val_dataloader.dataset)}")
        
        print("Setting up optimizer and scheduler...")
        optimizer, lr_scheduler = create_optimizer_scheduler(models, config)
        
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
        
        print("Setup completed successfully!")
        return {
            'config': config,
            'models': models, 
            'tokenizers': tokenizers,
            'train_dataloader': train_dataloader,
            'val_dataloader': val_dataloader,
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'device': device,
            'scheduler': scheduler
        }
        
    except Exception as e:
        print(f"Setup failed: {e}")
        raise