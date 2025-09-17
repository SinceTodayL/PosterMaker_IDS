"""
    IDS_TextEmbedder
"""


import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)
from ..utils.ids_tokenizer import IDSTokenizer
from ..utils.text_utils import normalize_coordinates, pos2coords, get_positional_encoding


class IDSEmbedding(nn.Module):
    
    def __init__(self, vocab_size: int, embed_dim: int = 64, 
                 max_seq_length: int = 512, dropout: float = 0.1,
                 char2feat_path: Optional[str] = None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.token_type_embedding = nn.Embedding(2, embed_dim)  # structure vs component
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Load pre-trained character features for initialization
        self.char2feat = None
        if char2feat_path:
            try:
                self.char2feat = torch.load(char2feat_path, map_location='cpu')
                print(f"Loaded pre-trained character features from {char2feat_path}")
            except Exception as e:
                print(f"Warning: Could not load char2feat from {char2feat_path}: {e}")
                self.char2feat = None
        
        # Initialize embeddings properly for training
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize embeddings with char2feat alignment when available"""
        # Default Xavier initialization
        nn.init.xavier_uniform_(self.token_embedding.weight)
        nn.init.xavier_uniform_(self.token_type_embedding.weight)
        nn.init.xavier_uniform_(self.position_embedding.weight)
        
        # Set padding token to zero
        with torch.no_grad():
            self.token_embedding.weight[0].fill_(0)
            
        
    def set_char2feat_alignment(self, tokenizer, alignment_weight: float = 0.5):

        if self.char2feat is None:
            print("Nothing to do in set_char2feat_alignment()")
            return
            
        # print("Performing char2feat alignment...")
        aligned_count = 0
        
        with torch.no_grad():
            for token_id, token in tokenizer.id_to_token.items():
                if token in self.char2feat and token_id > 0:  # Skip special tokens
                    # Get pre-trained feature
                    pretrained_feat = self.char2feat[token]  # Shape: (64,)
                    
                    # Blend with current embedding
                    current_embedding = self.token_embedding.weight[token_id]
                    blended_embedding = (
                        (1 - alignment_weight) * current_embedding + 
                        alignment_weight * pretrained_feat
                    )
                    
                    self.token_embedding.weight[token_id] = blended_embedding
                    aligned_count += 1
                    
        print(f"Successfully aligned {aligned_count} tokens with pre-trained features")
        
    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        position_ids = torch.arange(seq_len, device=device).expand(batch_size, -1)
        
        token_embeds = self.token_embedding(input_ids)
        type_embeds = self.token_type_embedding(token_type_ids)
        pos_embeds = self.position_embedding(position_ids)
        
        # Combine
        embeddings = token_embeds + type_embeds + pos_embeds
        
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        embeddings = embeddings * attention_mask.unsqueeze(-1)
        
        return embeddings


class FourierEmbedder:
    """Fourier positional embedding for spatial coordinates"""
    
    def __init__(self, num_freqs=8, temperature=100):
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


class IDSStructuralAttention(nn.Module):
    """Structural attention for IDS sequences to capture hierarchical relationships"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head attention for structural understanding
        self.structural_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(0.1)
        )
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

        attn_mask = attention_mask  
        attn_mask = attn_mask.float()
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
        attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
        
        attn_out, _ = self.structural_attention(x, x, x, key_padding_mask=(attention_mask == 0))
        x = self.layer_norm1(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)
        
        return x


class IDSTextEmbedder(nn.Module):
    
    def __init__(self, ids_database_path: str, vocab_file: Optional[str] = None,
                 ids_embed_dim: int = 64, max_seq_length: int = 128,
                 max_num_texts: int = 7, max_chars_per_text: int = 16,
                 input_size: tuple = (1024, 1024),
                 use_structural_attention: bool = True,
                 char2feat_path: Optional[str] = None,
                 char2feat_alignment_weight: float = 0.5):
        super().__init__()
        
        self.tokenizer = IDSTokenizer(
            ids_database_path=ids_database_path,
            vocab_file=vocab_file,
            preserve_rare_chars=True
        )
        self.ids_embedding = IDSEmbedding(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=ids_embed_dim,
            max_seq_length=max_seq_length,
            char2feat_path=char2feat_path  
        )
        
        if char2feat_alignment_weight > 0 and char2feat_path:
            self.ids_embedding.set_char2feat_alignment(
                self.tokenizer, char2feat_alignment_weight
            )
        
        self.use_structural_attention = use_structural_attention
        if use_structural_attention:
            self.structural_attention = IDSStructuralAttention(
                embed_dim=ids_embed_dim, num_heads=4
            )
        
        self.max_num_texts = max_num_texts
        self.char_padding_to_len = max_chars_per_text  # 16
        self.char_pos_encoding_dim = 32
        self.text_pos_encoding_dim = 32
        self.input_size = input_size
        
        self.fourier_embedder = FourierEmbedder(num_freqs=self.text_pos_encoding_dim // (4*2))
        
        # ids_embed_dim (64) + char_pos_encoding_dim (32) + text_pos_encoding_dim (32) = 128
        assert ids_embed_dim + self.char_pos_encoding_dim + self.text_pos_encoding_dim == 128, \
            "Feature dimensions must sum to 128 to match original architecture"
            
        self.sequence_pooler = nn.Sequential(
            nn.Linear(ids_embed_dim, ids_embed_dim),
            nn.Tanh()
        )
        
    def encode_single_text(self, text_content: str, max_length: int = None) -> Dict[str, torch.Tensor]:
        """Encode a single text string to IDS tokens"""
        if max_length is None:
            max_length = 64  
            
        encoding = self.tokenizer.encode_text(
            text_content, 
            max_length=max_length,
            add_special_tokens=True,
            use_recursive=False,
            enhance_rare_chars=False,
        )
        
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(encoding['token_type_ids'], dtype=torch.long)
        }
    
    def _pool_ids_sequence(self, ids_embeddings: torch.Tensor, 
                          attention_mask: torch.Tensor) -> tuple:
        """
        Pool variable-length IDS sequence to fixed length
        Args:
            ids_embeddings: (seq_len, embed_dim)
            attention_mask: (seq_len,)
        Returns:
            pooled_features: (max_chars_per_text, embed_dim)
            pooled_mask: (max_chars_per_text,)
        """
        seq_len, embed_dim = ids_embeddings.shape
        
        if seq_len <= self.char_padding_to_len:
            padding = torch.zeros(
                self.char_padding_to_len - seq_len, embed_dim, 
                device=ids_embeddings.device, dtype=ids_embeddings.dtype
            )
            pooled_features = torch.cat([ids_embeddings, padding], dim=0)
            
            mask_padding = torch.zeros(
                self.char_padding_to_len - seq_len,
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            pooled_mask = torch.cat([attention_mask, mask_padding], dim=0)
        else:
            stride = seq_len / self.char_padding_to_len
            pooled_features = []
            pooled_mask = []
            
            for i in range(self.char_padding_to_len):
                start_idx = int(i * stride)
                end_idx = min(int((i + 1) * stride), seq_len)
                
                if start_idx < end_idx:
                    segment_features = ids_embeddings[start_idx:end_idx]
                    segment_mask = attention_mask[start_idx:end_idx].float()
                    
                    if torch.sum(segment_mask) > 0:
                        weighted_features = torch.sum(
                            segment_features * segment_mask.unsqueeze(-1), dim=0
                        ) / torch.sum(segment_mask)
                        pooled_features.append(weighted_features)
                        pooled_mask.append(1.0)
                    else:
                        pooled_features.append(torch.zeros(embed_dim, device=ids_embeddings.device))
                        pooled_mask.append(0.0)
                else:
                    pooled_features.append(torch.zeros(embed_dim, device=ids_embeddings.device))
                    pooled_mask.append(0.0)
            
            pooled_features = torch.stack(pooled_features, dim=0)  # (16, embed_dim)
            pooled_mask = torch.tensor(pooled_mask, device=attention_mask.device)  # (16,)
        
        return pooled_features, pooled_mask
    
    def forward(self, texts: List[Dict[str, Any]]) -> torch.Tensor:

        device = next(self.parameters()).device
        
        # Process each text to get IDS features
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
            
            # 3. Apply structural attention if enabled
            if self.use_structural_attention:
                ids_embeddings = self.structural_attention(
                    ids_embeddings.unsqueeze(0), 
                    attention_mask.unsqueeze(0)
                ).squeeze(0)
            
            # 4. Pool sequence to fixed length
            char_features, char_token_mask = self._pool_ids_sequence(
                ids_embeddings, attention_mask
            )  # (16, 64), (16,)
            
            text_features.append(char_features)
            ocr_token_masks.append(char_token_mask)
        
        # 5. Add positional encodings (same as original TextEmbedder)
        
        # Character-level positional encoding
        char_positional_encoding = get_positional_encoding(
            self.char_padding_to_len, self.char_pos_encoding_dim
        ).to(device)  # (16, 32)
        
        for i in range(len(text_features)):
            text_features[i] = torch.cat([
                text_features[i], 
                ocr_token_masks[i].unsqueeze(-1) * char_positional_encoding
            ], dim=-1)  # (16, 64+32=96)
        
        # Text-level positional encoding
        for i in range(len(text_features)):
            coords = pos2coords(texts[i]['pos'])  # xyxy -> xywh
            coords_norm = torch.tensor(
                normalize_coordinates(coords, self.input_size[1], self.input_size[0]), 
                device=device
            )
            text_coords_embed = self.fourier_embedder(coords_norm)  # 32
            text_coords_embed = text_coords_embed.unsqueeze(0).repeat(
                self.char_padding_to_len, 1
            )  # (16, 32)
            text_features[i] = torch.cat([
                text_features[i], 
                ocr_token_masks[i].unsqueeze(-1) * text_coords_embed
            ], dim=-1)  # (16, 96+32=128)
        
        # 6. Assemble final output (matching original TextEmbedder exactly)
        max_token_num = self.char_padding_to_len * self.max_num_texts  # 16*7=112
        padding_token_num = max_token_num - self.char_padding_to_len * len(text_features)
        texts_and_sep_list = []
        
        for i in range(len(text_features)):
            texts_and_sep_list.append(text_features[i])
        
        # Add padding
        if padding_token_num > 0:
            texts_and_sep_list.append(
                torch.zeros((padding_token_num, 128), device=device)
            )
        
        texts_all_features = torch.cat(texts_and_sep_list, dim=0)  # (, 128)
        
        return texts_all_features  # (, 128) - correct output shape
    
    def get_text_embeds_batch(self, batch_texts: List[List[Dict[str, Any]]]) -> torch.Tensor:

        batch_embeds = []
        for texts in batch_texts:
            text_embeds = self.forward(texts)
            batch_embeds.append(text_embeds)

        batch_embeds = torch.stack(batch_embeds, dim=0)  # (batch_size, , 128)
        return batch_embeds
    
    def forward_tokenized_batch(self, input_ids: torch.Tensor, 
                               attention_mask: torch.Tensor,
                               token_type_ids: torch.Tensor,
                               text_pos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for batch of tokenized IDS sequences.
        Fixed to maintain token-level structure matching original TextEmbedder.
        
        Args:
            input_ids: (batch_size, seq_len) tokenized IDS sequences
            attention_mask: (batch_size, seq_len) attention mask
            token_type_ids: (batch_size, seq_len) token type IDs
            text_pos: (batch_size, 4) normalized text positions [x1, y1, x2, y2]
            
        Returns:
            torch.Tensor: (batch_size, 112, 128) token-level features for ControlNet
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 1. Get IDS embeddings from tokenized input
        ids_embeddings = self.ids_embedding(input_ids, token_type_ids, attention_mask)  # (batch_size, seq_len, embed_dim)
        
        # 2. Apply structural attention if enabled
        if self.use_structural_attention:
            ids_embeddings = self.structural_attention(ids_embeddings, attention_mask)  # (batch_size, seq_len, embed_dim)
        
        # 3. Reshape to fixed token structure
        # Pool/reshape the variable-length sequence to fixed 16 tokens per text
        batch_features = []
        
        for batch_idx in range(batch_size):
            # Get single sequence
            seq_embeddings = ids_embeddings[batch_idx]  # (seq_len, 64)
            seq_mask = attention_mask[batch_idx]  # (seq_len,)
            
            # Pool to fixed length (16 tokens)
            pooled_seq, _ = self._pool_ids_sequence(seq_embeddings, seq_mask)  # (16, 64)
            
            # Add character-level positional encoding
            char_pos_encoding = get_positional_encoding(
                self.char_padding_to_len, self.char_pos_encoding_dim
            ).to(device)  # (16, 32)
            
            # Combine with char positional encoding
            char_features = torch.cat([
                pooled_seq,  # (16, 64)
                char_pos_encoding  # (16, 32)
            ], dim=-1)  # (16, 96)
            
            # Add text-level positional encoding with overflow protection
            try:
                # Clamp values to prevent overflow and convert safely
                text_pos_clamped = torch.clamp(text_pos[batch_idx], min=-1e6, max=1e6)
                coords = pos2coords(text_pos_clamped.cpu().numpy().astype(np.float32))  # Convert to xywh
                coords_norm = normalize_coordinates(coords, self.input_size[1], self.input_size[0])
                text_coords_embed = self.fourier_embedder(torch.tensor(coords_norm, device=device, dtype=torch.float32))  # (32,)
            except (OverflowError, ValueError) as e:
                # Fallback to zero coordinates if conversion fails
                logger.warning(f"Position encoding overflow for batch {batch_idx}, using zero coordinates: {e}")
                text_coords_embed = torch.zeros(32, device=device, dtype=torch.float32)
            text_coords_embed = text_coords_embed.unsqueeze(0).repeat(self.char_padding_to_len, 1)  # (16, 32)
            
            # Final features for this text
            text_features = torch.cat([
                char_features,  # (16, 96)
                text_coords_embed  # (16, 32)
            ], dim=-1)  # (16, 128)
            
            batch_features.append(text_features)
        
        # 4. Assemble to match original TextEmbedder output format
        batch_outputs = []
        for batch_idx in range(batch_size):
            # Create output matching (112, 128) structure
            max_token_num = self.char_padding_to_len * self.max_num_texts  # 16 * 7 = 112
            
            # Use single text (can be extended for multiple texts later)
            text_features = batch_features[batch_idx]  # (16, 128)
            
            # Pad to full 112 tokens
            padding_token_num = max_token_num - self.char_padding_to_len  # 112 - 16 = 96
            padding_features = torch.zeros((padding_token_num, 128), device=device)
            
            full_features = torch.cat([text_features, padding_features], dim=0)  # (112, 128)
            batch_outputs.append(full_features)
        
        # Stack batch
        batch_tensor = torch.stack(batch_outputs, dim=0)  # (batch_size, 112, 128)
        
        return batch_tensor

"""
    def get_lora_target_modules(self) -> List[str]:
        # Return module names that should be targeted by LoRA
        lora_targets = [
            # IDS embedding layers
            "ids_embedding.token_embedding",
            "ids_embedding.token_type_embedding", 
            "ids_embedding.position_embedding",
            
            # Structural attention layers (if enabled)
            "structural_attention.structural_attention.in_proj_weight",
            "structural_attention.structural_attention.out_proj",
            "structural_attention.ffn.0",  # First linear layer in FFN
            "structural_attention.ffn.3",  # Second linear layer in FFN
            
            # Sequence pooler
            "sequence_pooler.0"
        ]
        
        if not self.use_structural_attention:
            # Remove structural attention targets if not used
            lora_targets = [target for target in lora_targets 
                          if not target.startswith("structural_attention")]
        
        return lora_targets
"""