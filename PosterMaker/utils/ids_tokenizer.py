"""
IDS Tokenizer for converting IDS sequences to token arrays
Supports both structural descriptors (⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻) and components
"""

import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import torch
from utils.ids_query import IDSQuery


class IDSTokenizer:
    """
    Tokenizer for IDS (Ideographic Description Sequences)
    Converts IDS sequences to token arrays suitable for neural networks
    """
    
    def __init__(self, vocab_file: Optional[str] = r'.\assets\ids_vocab.json', build_vocab: bool = False):
        """
        Initialize IDS tokenizer
        
        Args:
            vocab_file: Path to existing vocabulary file
            build_vocab: Whether to build vocabulary from IDS data
        """
        self.ids_query = IDSQuery()
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<BOS>': 1, 
            '<EOS>': 2,
            '<UNK>': 3,
            '<SEP>': 4,  # Separator between characters
        }
        
        # IDS structural descriptors (12 total)
        self.idc_chars = {
            '⿰', '⿱', '⿲', '⿳', '⿴', '⿵', '⿶', '⿷', '⿸', '⿹', '⿺', '⿻'
        }
        
        # test if vacal_file exists
        # print(Path(vocab_file).resolve())

        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0
        
        if vocab_file and Path(vocab_file).exists():
            self.load_vocab(vocab_file)
        elif build_vocab:
            self.build_vocabulary()

    def build_vocabulary(self):
        """
        Build vocabulary from IDS database
        Order: special_tokens -> idc_chars -> components
        """
        print("Building IDS vocabulary...")
        
        # Start with special tokens
        self.token_to_id = self.special_tokens.copy()
        
        # Add IDS structural descriptors
        current_id = len(self.special_tokens)
        for idc in sorted(self.idc_chars):
            self.token_to_id[idc] = current_id
            current_id += 1
            
        # Collect all unique components from IDS database
        all_components = set()
        
        for char, ids_list in self.ids_query.char_to_ids.items():
            # Add the character itself
            all_components.add(char)
            
            # Add all components from its IDS representations
            for ids in ids_list:
                for component in ids:
                    if component not in self.idc_chars:  # Skip structural descriptors
                        all_components.add(component)
        
        # Add components to vocabulary (sorted for consistency)
        for component in sorted(all_components):
            if component not in self.token_to_id:
                self.token_to_id[component] = current_id
                current_id += 1
                
        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)
        
        print(f"Built vocabulary with {self.vocab_size} tokens:")
        print(f"   Special tokens: {len(self.special_tokens)}")
        print(f"   IDC chars: {len(self.idc_chars)}")
        print(f"   Components: {self.vocab_size - len(self.special_tokens) - len(self.idc_chars)}")
        
    def save_vocab(self, vocab_file: str):
        """Save vocabulary to file"""
        vocab_data = {
            'token_to_id': self.token_to_id,
            'special_tokens': self.special_tokens,
            'idc_chars': list(self.idc_chars),
            'vocab_size': self.vocab_size
        }
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved to {vocab_file}")
        
    def load_vocab(self, vocab_file: str):
        """Load vocabulary from file"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            
        self.token_to_id = vocab_data['token_to_id']
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = vocab_data['vocab_size']
        self.special_tokens = vocab_data['special_tokens']
        self.idc_chars = set(vocab_data['idc_chars'])
        
        print(f"Loaded vocabulary with {self.vocab_size} tokens from {vocab_file}")
        


    def encode_char(self, char: str, use_recursive: bool = False) -> List[int]:
        """
        Encode a single character to IDS token sequence
        
        Args:
            char: Chinese character to encode
            use_recursive: Whether to use recursive decomposition
            
        Returns:
            List of token IDs
        """
        if char not in self.ids_query.char_to_ids:
            # Character not in IDS database, return as single token
            if char in self.token_to_id:
                return [self.token_to_id[char]]
            else:
                return [self.special_tokens['<UNK>']]
        
        if use_recursive:
            # Use recursive decomposition
            result = self.ids_query._query_recursive(char)
            ids_seq = self.ids_query._flatten_recursive_result(result)
        else:
            # Use direct IDS representation (first one)
            ids_list = self.ids_query._query_direct(char)
            ids_seq = ids_list[0] if ids_list else char
            
        # Convert IDS sequence to token IDs
        token_ids = []
        for token in ids_seq:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.special_tokens['<UNK>'])
                
        return token_ids
    
    def encode_text(self, text: str, max_length: Optional[int] = None, 
                   add_special_tokens: bool = True, use_recursive: bool = False) -> Dict:
        """
        Encode text to IDS token sequence
        
        Args:
            text: Input text string
            max_length: Maximum sequence length (truncate if exceeded)
            add_special_tokens: Whether to add BOS/EOS tokens
            use_recursive: Whether to use recursive decomposition
            
        Returns:
            Dictionary containing:
            - input_ids: token IDs
            - attention_mask: attention mask (1 for real tokens, 0 for padding)
            - token_type_ids: token type (0 for structure, 1 for component)
        """
        all_token_ids = []
        all_token_types = []  # 0 for IDC, 1 for component
        
        if add_special_tokens:
            all_token_ids.append(self.special_tokens['<BOS>'])
            all_token_types.append(0)
            
        for i, char in enumerate(text):
            if '\u4e00' <= char <= '\u9fff':  # Chinese characters only
                char_tokens = self.encode_char(char, use_recursive)
                all_token_ids.extend(char_tokens)
                
                # Determine token types
                for token_id in char_tokens:
                    token = self.id_to_token[token_id]
                    if token in self.idc_chars:
                        all_token_types.append(0)  # Structure
                    else:
                        all_token_types.append(1)  # Component
                        
                # Add separator between characters (except last)
                if i < len(text) - 1:
                    all_token_ids.append(self.special_tokens['<SEP>'])
                    all_token_types.append(0)
            else:
                # Non-Chinese characters - encode as single token
                if char in self.token_to_id:
                    all_token_ids.append(self.token_to_id[char])
                else:
                    all_token_ids.append(self.special_tokens['<UNK>'])
                all_token_types.append(1)
                
        if add_special_tokens:
            all_token_ids.append(self.special_tokens['<EOS>'])
            all_token_types.append(0)
            
        # Truncate if necessary
        if max_length and len(all_token_ids) > max_length:
            all_token_ids = all_token_ids[:max_length-1] + [self.special_tokens['<EOS>']]
            all_token_types = all_token_types[:max_length-1] + [0]
            
        # Create attention mask
        attention_mask = [1] * len(all_token_ids)
        
        # Pad if max_length specified
        if max_length:
            padding_length = max_length - len(all_token_ids)
            if padding_length > 0:
                all_token_ids.extend([self.special_tokens['<PAD>']] * padding_length)
                all_token_types.extend([0] * padding_length)
                attention_mask.extend([0] * padding_length)
                
        return {
            'input_ids': all_token_ids,
            'attention_mask': attention_mask,
            'token_type_ids': all_token_types
        }
    
    def encode_text_batch(self, texts: List[str], max_length: Optional[int] = None,
                         add_special_tokens: bool = True, use_recursive: bool = False) -> Dict:
        """
        Encode batch of texts
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length  
            add_special_tokens: Whether to add BOS/EOS tokens
            use_recursive: Whether to use recursive decomposition
            
        Returns:
            Dictionary with batched encodings
        """
        batch_encodings = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': []
        }
        
        for text in texts:
            encoding = self.encode_text(text, max_length, add_special_tokens, use_recursive)
            batch_encodings['input_ids'].append(encoding['input_ids'])
            batch_encodings['attention_mask'].append(encoding['attention_mask'])
            batch_encodings['token_type_ids'].append(encoding['token_type_ids'])
            
        return batch_encodings
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.special_tokens.values():
                    continue
                tokens.append(token)
                
        return ''.join(tokens)
    
    def get_vocab_info(self) -> Dict:
        """Get vocabulary statistics"""
        idc_count = sum(1 for token in self.token_to_id.keys() if token in self.idc_chars)
        component_count = self.vocab_size - len(self.special_tokens) - idc_count
        
        return {
            'total_vocab_size': self.vocab_size,
            'special_tokens': len(self.special_tokens),
            'idc_chars': idc_count,
            'components': component_count,
            'special_token_names': list(self.special_tokens.keys()),
            'idc_chars_list': sorted(list(self.idc_chars))
        } 