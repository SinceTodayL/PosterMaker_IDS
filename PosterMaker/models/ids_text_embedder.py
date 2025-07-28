"""
IDS-based Text Embedder for TextRenderNet
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from utils.ids_tokenizer import IDSTokenizer
from utils.utils import normalize_coordinates, pos2coords, get_positional_encoding


class IDSEmbedding(nn.Module):
    """IDS Token Embedding with positional and structural encoding"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 256, 
                 max_seq_length: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Token type embedding (structure vs component)
        self.token_type_embedding = nn.Embedding(2, embed_dim)
        # Position embedding within sequence
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize embeddings with proper scaling"""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.token_type_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.02)
        
        # Set padding token to zero
        with torch.no_grad():
            self.token_embedding.weight[0].fill_(0)
            
    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of IDS embedding"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Position IDs
        position_ids = torch.arange(seq_len, device=device).expand(batch_size, -1)
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        type_embeds = self.token_type_embedding(token_type_ids)
        pos_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeddings = token_embeds + type_embeds + pos_embeds
        
        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Apply attention mask
        embeddings = embeddings * attention_mask.unsqueeze(-1)
        
        return embeddings


class FourierEmbedder:
    """Fourier positional embedding for spatial coordinates"""
    
    def __init__(self, num_freqs=64, temperature=100):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)

    @torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        """x: arbitrary shape of tensor. dim: cat dim"""
        out = []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, cat_dim)


class IDSTextEmbedder(nn.Module):
    """IDS-based Text Embedder for PosterMaker TextRenderNet"""
    
    def __init__(self, vocab_file: str = './assets/ids_vocab.json',
                 ids_embed_dim: int = 256, max_seq_length: int = 512,
                 max_num_texts: int = 7, max_chars_per_text: int = 16,
                 input_size: tuple = (1024, 1024)):
        super().__init__()
        
        # IDS tokenizer and embedding
        self.tokenizer = IDSTokenizer(vocab_file=vocab_file, build_vocab=False)
        self.ids_embedding = IDSEmbedding(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=ids_embed_dim,
            max_seq_length=max_seq_length
        )
        
        # Text layout parameters
        self.max_num_texts = max_num_texts
        self.char_padding_to_len = max_chars_per_text  # 16
        self.char_pos_encoding_dim = 32
        self.text_pos_encoding_dim = 32
        self.input_size = input_size
        
        # Spatial positional encoding
        self.fourier_embedder = FourierEmbedder(num_freqs=self.text_pos_encoding_dim // (4*2))
        
        # Feature projection
        self.feature_projector = nn.Linear(ids_embed_dim, 64)  # Project IDS features to 64-dim
        
    def encode_single_text(self, text_content: str, max_length: int = None) -> Dict[str, torch.Tensor]:
        """Encode a single text string to IDS tokens"""
        if max_length is None:
            max_length = 128
            
        encoding = self.tokenizer.encode_text(
            text_content, 
            max_length=max_length,
            add_special_tokens=True,
            use_recursive=False
        )
        
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(encoding['token_type_ids'], dtype=torch.long)
        }
    
    def forward(self, texts: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Forward pass of IDS Text Embedder
        """
        device = next(self.parameters()).device
        
        # Process each text to get IDS feature list
        text_features = []
        ocr_token_masks = []
        
        for text_info in texts:
            content = text_info['content']
            
            # 1. Encode to IDS tokens
            encoding = self.encode_single_text(content)
            
            # Move to device
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            token_type_ids = encoding['token_type_ids'].to(device)
            
            # 2. Get IDS embeddings
            ids_embeddings = self.ids_embedding(
                input_ids.unsqueeze(0), 
                token_type_ids.unsqueeze(0), 
                attention_mask.unsqueeze(0)
            )  # (1, seq_len, embed_dim)
            ids_embeddings = ids_embeddings.squeeze(0)  # (seq_len, embed_dim)
            
            # 3. Project to 64-dim (corresponding to original OCR features)
            ids_features = self.feature_projector(ids_embeddings)  # (seq_len, 64)
            
            # 4. Compress to 16 positions (matching original char_padding_to_len)
            actual_length = ids_features.shape[0]
            if actual_length > self.char_padding_to_len:
                # Use weighted average pooling to compress long sequences
                segment_size = actual_length / self.char_padding_to_len
                compressed_features = []
                compressed_mask = []
                
                for i in range(self.char_padding_to_len):
                    start_idx = int(i * segment_size)
                    end_idx = min(int((i + 1) * segment_size), actual_length)
                    
                    if start_idx < end_idx:
                        segment_features = ids_features[start_idx:end_idx]
                        segment_mask = attention_mask[start_idx:end_idx].float()
                        
                        if torch.sum(segment_mask) > 0:
                            weighted_features = torch.sum(segment_features * segment_mask.unsqueeze(-1), dim=0) / torch.sum(segment_mask)
                            compressed_features.append(weighted_features)
                            compressed_mask.append(1.0)
                        else:
                            compressed_features.append(torch.zeros(64, device=device))
                            compressed_mask.append(0.0)
                    else:
                        compressed_features.append(torch.zeros(64, device=device))
                        compressed_mask.append(0.0)
                
                char_features = torch.stack(compressed_features, dim=0)  # (16, 64)
                char_token_mask = torch.tensor(compressed_mask, device=device)  # (16,)
            else:
                # Length <= 16, directly pad
                if actual_length < self.char_padding_to_len:
                    padding = torch.zeros(self.char_padding_to_len - actual_length, 64, device=device)
                    char_features = torch.cat([ids_features, padding], dim=0)
                    
                    padding_mask = torch.zeros(self.char_padding_to_len - actual_length, device=device)
                    char_token_mask = torch.cat([attention_mask.float(), padding_mask], dim=0)
                else:
                    char_features = ids_features[:self.char_padding_to_len]
                    char_token_mask = attention_mask[:self.char_padding_to_len].float()
            
            text_features.append(char_features)
            ocr_token_masks.append(char_token_mask)
        
        # 5. Add positional encoding following original TextEmbedder flow
        pos_dim = self.char_pos_encoding_dim + self.text_pos_encoding_dim
        feature_dim = text_features[0].shape[-1]  # 64
        
        # Character-level positional encoding
        char_positional_encoding = get_positional_encoding(self.char_padding_to_len, self.char_pos_encoding_dim)
        char_positional_encoding = char_positional_encoding.to(device)  # (16, 32)
        
        for i in range(len(text_features)):
            text_features[i] = torch.cat([
                text_features[i], 
                ocr_token_masks[i].unsqueeze(-1) * char_positional_encoding
            ], dim=-1)  # (16, 64+32)
        
        # Text-level positional encoding
        for i in range(len(text_features)):
            coords = pos2coords(texts[i]['pos'])  # xyxy -> xywh
            coords_norm = torch.tensor(normalize_coordinates(coords, self.input_size[1], self.input_size[0]), device=device)
            text_coords_embed = self.fourier_embedder(coords_norm)  # 32
            text_coords_embed = text_coords_embed.unsqueeze(0).repeat(self.char_padding_to_len, 1)  # (16, 32)
            text_features[i] = torch.cat([
                text_features[i], 
                ocr_token_masks[i].unsqueeze(-1) * text_coords_embed
            ], dim=-1)  # (16, 64+32+32=128)
        
        # 6. Assemble final output (matching original TextEmbedder exactly)
        max_token_num = self.char_padding_to_len * self.max_num_texts  # 16*7=112
        padding_token_num = max_token_num - self.char_padding_to_len * len(text_features)
        texts_and_sep_list = []
        
        for i in range(len(text_features)):
            texts_and_sep_list.append(text_features[i])
        
        # Add padding
        if padding_token_num > 0:
            texts_and_sep_list.append(torch.zeros((padding_token_num, pos_dim + feature_dim), device=device))
        
        texts_all_features = torch.cat(texts_and_sep_list, dim=0)  # (112, 128)
        
        # CRITICAL: Expand to 1472 dimensions to match original training expectations
        # Based on wrapper_models.py comment: text embed shape: [b, 128, 1472]
        batch_size = 1 if len(texts_all_features.shape) == 2 else texts_all_features.shape[0]
        seq_len = texts_all_features.shape[-2] if len(texts_all_features.shape) == 3 else texts_all_features.shape[0]
        
        # Add feature expansion to reach 1472 dimensions
        # This could be achieved through learned projection or feature augmentation
        expansion_dim = 1472 - 128
        
        # Simple approach: learn a projection from 128 to 1472
        if not hasattr(self, 'feature_expander'):
            self.feature_expander = nn.Linear(128, 1472).to(texts_all_features.device)
            
        # Expand features
        if len(texts_all_features.shape) == 2:
            expanded_features = self.feature_expander(texts_all_features)  # (112, 1472)
        else:
            expanded_features = self.feature_expander(texts_all_features)  # (batch, 112, 1472)
        
        return expanded_features
    
    def get_text_embeds_batch(self, batch_texts: List[List[Dict[str, Any]]]) -> torch.Tensor:
        """Process batch of text lists"""
        batch_embeds = []
        for texts in batch_texts:
            text_embeds = self.forward(texts)
            batch_embeds.append(text_embeds)

        batch_embeds = torch.stack(batch_embeds, dim=0)
        return batch_embeds 