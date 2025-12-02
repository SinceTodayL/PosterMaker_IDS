"""
Character-level IDS Embedder
Processes each character independently: char -> IDS decomposition -> component embeddings -> attention -> pooling -> single vector
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ids_query import IDSQuery
from utils.ids_tokenizer import IDSTokenizer

logger = logging.getLogger(__name__)


class IDSComponentEmbedding(nn.Module):
    """Embedding layer for IDS components"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 64, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with Xavier uniform"""
        nn.init.xavier_uniform_(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight[self.embedding.padding_idx].fill_(0)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len) or (seq_len,)
        Returns:
            embeddings: (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
        """
        return self.embedding(input_ids)


class IDSComponentAttention(nn.Module):
    """Lightweight attention for IDS component sequences"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
            attention_mask: (batch_size, seq_len) or (seq_len,), 1 for valid, 0 for padding
        Returns:
            output: same shape as x
        """
        # Handle both batched and unbatched inputs
        is_unbatched = x.dim() == 2
        if is_unbatched:
            x = x.unsqueeze(0)  # (1, seq_len, embed_dim)
        
        # Prepare attention mask
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            # Convert to key_padding_mask format (True for padding, False for valid)
            attn_mask = (attention_mask == 0)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
        
        # Residual connection and layer norm
        x = self.layer_norm(x + self.dropout(attn_out))
        
        if is_unbatched:
            x = x.squeeze(0)  # (seq_len, embed_dim)
        
        return x


class CharIDSEmbedder(nn.Module):
    """
    Character-level IDS embedder
    Processes each character independently to produce a single 64-dim vector
    """
    
    def __init__(self, 
                 ids_database_path: str,
                 vocab_file: Optional[str] = None,
                 ids_embed_dim: int = 64,
                 max_ids_seq_length: int = 32,
                 use_attention: bool = True,
                 num_attention_heads: int = 4):
        super().__init__()
        
        self.ids_embed_dim = ids_embed_dim
        self.max_ids_seq_length = max_ids_seq_length
        self.use_attention = use_attention
        
        # Initialize IDS query and tokenizer
        self.ids_query = IDSQuery(ids_database_path)
        self.tokenizer = IDSTokenizer(
            ids_database_path=ids_database_path,
            vocab_file=vocab_file,
            preserve_rare_chars=True
        )
        
        # Component embedding layer
        self.component_embedding = IDSComponentEmbedding(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=ids_embed_dim,
            padding_idx=self.tokenizer.special_tokens['<PAD>']
        )
        
        # Attention layer (optional)
        if use_attention:
            self.attention = IDSComponentAttention(
                embed_dim=ids_embed_dim,
                num_heads=num_attention_heads
            )
        else:
            self.attention = None
        
        # Average pooling (always used to get single vector)
        # No learnable parameters, just functional pooling
    
    def _get_ids_sequence(self, char: str) -> list:
        """
        Get IDS sequence for a character (simple decomposition, not recursive)
        
        Args:
            char: Single Chinese character
            
        Returns:
            List of IDS components (characters and IDCs)
        """
        if not char or len(char) != 1:
            return []
        
        # Query IDS decomposition (simple, not recursive)
        ids_list = self.ids_query._query_direct(char)
        if not ids_list:
            # Character not in database, return empty list
            return []
        
        # Use first IDS representation
        ids_seq = ids_list[0]
        
        # Convert to list of components
        components = []
        for component in ids_seq:
            components.append(component)
        
        return components
    
    def _encode_ids_sequence(self, ids_components: list) -> tuple:
        """
        Encode IDS component sequence to token IDs
        
        Args:
            ids_components: List of IDS components (strings)
            
        Returns:
            (token_ids, attention_mask)
            token_ids: List of token IDs
            attention_mask: List of 1s and 0s (1 for valid, 0 for padding)
        """
        if not ids_components:
            # Empty sequence: return padding
            return (
                [self.tokenizer.special_tokens['<PAD>']] * self.max_ids_seq_length,
                [0] * self.max_ids_seq_length
            )
        
        token_ids = []
        for component in ids_components:
            if component in self.tokenizer.token_to_id:
                token_ids.append(self.tokenizer.token_to_id[component])
            else:
                token_ids.append(self.tokenizer.special_tokens['<UNK>'])
        
        # Truncate if too long
        if len(token_ids) > self.max_ids_seq_length:
            token_ids = token_ids[:self.max_ids_seq_length]
        
        # Pad if too short
        attention_mask = [1] * len(token_ids)
        padding_length = self.max_ids_seq_length - len(token_ids)
        if padding_length > 0:
            token_ids.extend([self.tokenizer.special_tokens['<PAD>']] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        return token_ids, attention_mask
    
    def encode_char(self, char: str, return_debug_info: bool = False, print_decomposition: bool = False) -> torch.Tensor:
        """
        Encode a single character to IDS feature vector
        
        Args:
            char: Single Chinese character
            return_debug_info: If True, return tuple (feature, debug_info)
            print_decomposition: If True, print character decomposition info
            
        Returns:
            feature_vector: (ids_embed_dim,) tensor, 64-dim vector
            or (feature_vector, debug_info) if return_debug_info=True
        """
        device = next(self.parameters()).device
        
        # Get IDS decomposition
        ids_components = self._get_ids_sequence(char)
        
        # Print decomposition if requested
        if print_decomposition:
            if ids_components:
                ids_str = ''.join(ids_components)
                logger.info(f"  [IDS Decomposition] Character: '{char}' -> IDS sequence: {ids_str} (components: {len(ids_components)})")
            else:
                logger.info(f"  [IDS Decomposition] Character: '{char}' -> No IDS decomposition found (using fallback)")
        
        # Encode to token IDs
        token_ids, attention_mask = self._encode_ids_sequence(ids_components)
        
        debug_info = None
        if return_debug_info:
            debug_info = {
                'char': char,
                'ids_components': ids_components,
                'num_components': len(ids_components),
                'token_ids': token_ids[:len(ids_components)] if ids_components else []
            }
        
        # Convert to tensors
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)  # (seq_len,)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long, device=device)  # (seq_len,)
        
        # Get component embeddings
        component_embeds = self.component_embedding(token_ids_tensor)  # (seq_len, embed_dim)
        
        # Apply attention if enabled
        if self.attention is not None:
            component_embeds = self.attention(component_embeds, attention_mask_tensor)  # (seq_len, embed_dim)
        
        # Average pooling (weighted by attention mask)
        mask_float = attention_mask_tensor.float().unsqueeze(-1)  # (seq_len, 1)
        masked_embeds = component_embeds * mask_float  # (seq_len, embed_dim)
        
        # Sum and normalize by number of valid components
        mask_sum = mask_float.sum()
        if mask_sum > 1e-8:
            char_feature = masked_embeds.sum(dim=0) / mask_sum  # (embed_dim,)
        else:
            # Fallback: use zero vector if no valid components
            char_feature = torch.zeros(self.ids_embed_dim, device=device)
        
        if return_debug_info and debug_info is not None:
            debug_info['feature_norm'] = torch.norm(char_feature).item()
            debug_info['valid_components'] = int(mask_sum.item())
            return char_feature, debug_info
        
        return char_feature
    
    def forward(self, chars: list) -> torch.Tensor:
        """
        Encode a list of characters to feature vectors
        
        Args:
            chars: List of characters
            
        Returns:
            features: (len(chars), ids_embed_dim) tensor
        """
        features = []
        for char in chars:
            char_feat = self.encode_char(char)
            features.append(char_feat)
        
        return torch.stack(features, dim=0)  # (len(chars), ids_embed_dim)

