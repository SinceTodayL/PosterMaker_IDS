"""
IDS Tokenizer for converting IDS sequences to token arrays
Optimized for LoRA training with enhanced rare character support
Modified for configurable paths in training pipeline
"""

import json
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import torch
from .ids_query import IDSQuery

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IDSTokenizer:
    """
    Tokenizer for IDS (Ideographic Description Sequences)
    Converts IDS sequences to token arrays suitable for neural networks
    Optimized for LoRA training with enhanced rare character support
    """
    
    def __init__(self, ids_database_path: str, vocab_file: Optional[str] = None, 
                 build_vocab: bool = False, 
                 preserve_rare_chars: bool = True):
        """
        Initialize IDS tokenizer
        
        Args:
            ids_database_path: Path to IDS database file (ids_database.txt)
            vocab_file: Path to existing vocabulary file
            build_vocab: Whether to build vocabulary from IDS data
            preserve_rare_chars: Whether to preserve rare characters (recommended: True)
        """
        self.ids_query = IDSQuery(ids_database_path)
        
        # Special tokens with consistent IDs
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
        
        # Initialize empty mappings
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0
        self.preserve_rare_chars = preserve_rare_chars
        
        # Statistics for rare character analysis
        self.rare_char_stats = {
            'total_components': 0,
            'singleton_components': 0,  # Components appearing only once
            'rare_char_coverage': 0.0
        }
        
        # Load or build vocabulary
        if vocab_file and Path(vocab_file).exists():
            try:
                self.load_vocab(vocab_file)
                logger.info(f"Successfully loaded vocabulary from {vocab_file}")
            except Exception as e:
                logger.error(f"Failed to load vocabulary from {vocab_file}: {e}")
                if build_vocab:
                    logger.info("Building vocabulary from scratch...")
                    self.build_vocabulary()
                    if vocab_file:
                        self.save_vocab(vocab_file)
                else:
                    raise ValueError(f"Cannot load vocabulary and build_vocab=False: {e}")
        elif build_vocab:
            logger.info("Building vocabulary from IDS database...")
            self.build_vocabulary()
            if vocab_file:
                self.save_vocab(vocab_file)
        else:
            raise ValueError("Either provide valid vocab_file or set build_vocab=True")

    def build_vocabulary(self):
        """
        Build vocabulary from IDS database with ALL components preserved
        Enhanced support for rare characters - NO FILTERING
        Order: special_tokens -> idc_chars -> all_components
        """
        logger.info("Building comprehensive IDS vocabulary (preserving ALL characters)...")
        
        # Start with special tokens
        self.token_to_id = self.special_tokens.copy()
        
        # Add IDS structural descriptors
        current_id = len(self.special_tokens)
        for idc in sorted(self.idc_chars):
            self.token_to_id[idc] = current_id
            current_id += 1
            
        # Collect ALL components with frequency statistics (for analysis, not filtering)
        component_freq = {}
        
        for char, ids_list in self.ids_query.char_to_ids.items():
            # Add the character itself
            component_freq[char] = component_freq.get(char, 0) + 1
            
            # Add all components from its IDS representations
            for ids in ids_list:
                for component in ids:
                    if component not in self.idc_chars:  # Skip structural descriptors
                        component_freq[component] = component_freq.get(component, 0) + 1
        
        # Add ALL components to vocabulary (NO filtering based on frequency)
        all_components = set(component_freq.keys()) - set(self.token_to_id.keys())
        
        # Sort components: prioritize rare characters for better learning
        if self.preserve_rare_chars:
            # Sort by frequency (ascending): rare characters get lower IDs, potentially better embeddings
            sorted_components = sorted(all_components, key=lambda x: (component_freq.get(x, 0), x))
            logger.info("Vocabulary prioritizes rare characters (lower token IDs)")
        else:
            # Standard alphabetical sorting
            sorted_components = sorted(all_components)
        
        # Add all sorted components to vocabulary
        for component in sorted_components:
            self.token_to_id[component] = current_id
            current_id += 1
                
        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)
        
        # Calculate rare character statistics
        self.rare_char_stats['total_components'] = len(component_freq)
        self.rare_char_stats['singleton_components'] = sum(1 for freq in component_freq.values() if freq == 1)
        self.rare_char_stats['rare_char_coverage'] = self.rare_char_stats['singleton_components'] / self.rare_char_stats['total_components']
        
        logger.info(f"Built comprehensive vocabulary with {self.vocab_size} tokens:")
        logger.info(f"  Special tokens: {len(self.special_tokens)}")
        logger.info(f"  IDC chars: {len(self.idc_chars)}")
        logger.info(f"  Total components: {len(all_components)} (ALL preserved)")
        logger.info(f"  Rare characters (freq=1): {self.rare_char_stats['singleton_components']} ({self.rare_char_stats['rare_char_coverage']:.1%})")
        logger.info(f"  Coverage: Enhanced support for rare/complex characters")
        
    def save_vocab(self, vocab_file: str):
        """Save vocabulary to file with rare character metadata"""
        vocab_data = {
            'token_to_id': self.token_to_id,
            'special_tokens': self.special_tokens,
            'idc_chars': list(self.idc_chars),
            'vocab_size': self.vocab_size,
            'preserve_rare_chars': self.preserve_rare_chars,
            'rare_char_stats': self.rare_char_stats,
            'version': '1.2'  # Updated version for rare character support
        }
        
        vocab_path = Path(vocab_file)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Vocabulary with rare character support saved to {vocab_file}")
        
    def load_vocab(self, vocab_file: str):
        """Load vocabulary from file with rare character support"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            
        # Check version compatibility
        # version = vocab_data.get('version', '1.0')
        # if version not in ['1.1', '1.2']:
            # logger.warning(f"Loading vocabulary version {version}, current version is 1.2")
            
        self.token_to_id = vocab_data['token_to_id']
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}  # Ensure int keys
        self.vocab_size = vocab_data['vocab_size']
        self.special_tokens = vocab_data['special_tokens']
        self.idc_chars = set(vocab_data['idc_chars'])
        self.preserve_rare_chars = vocab_data.get('preserve_rare_chars', True)
        self.rare_char_stats = vocab_data.get('rare_char_stats', {})
        
        # Validate loaded vocabulary
        assert len(self.token_to_id) == self.vocab_size, \
            f"Vocabulary size mismatch: {len(self.token_to_id)} != {self.vocab_size}"
        assert len(self.id_to_token) == self.vocab_size, \
            f"Reverse mapping size mismatch: {len(self.id_to_token)} != {self.vocab_size}"
            
        logger.info(f"Loaded vocabulary with {self.vocab_size} tokens")
        if self.rare_char_stats:
            logger.info(f"  Rare character support: {self.rare_char_stats.get('singleton_components', 0)} rare chars")

    def encode_char(self, char: str, use_recursive: bool = False, 
                   prefer_complex_decomposition: bool = True) -> List[int]:
        """
        Encode a single character to IDS token sequence with rare character enhancement
        
        Args:
            char: Chinese character to encode
            use_recursive: Whether to use recursive decomposition
            prefer_complex_decomposition: Whether to prefer complex IDS for rare characters
            
        Returns:
            List of token IDs
        """
        # Handle empty or invalid input
        if not char or len(char) != 1:
            return [self.special_tokens['<UNK>']]
        
        # For rare characters, prefer recursive decomposition to expose more structure
        if prefer_complex_decomposition and char in self.ids_query.char_to_ids:
            ids_list = self.ids_query.char_to_ids[char]
            # Use recursive if character has complex structure (multiple IDS representations)
            if len(ids_list) > 1 or (ids_list and len(ids_list[0]) > 3):
                use_recursive = True
                
        # Check if character is in IDS database
        if char not in self.ids_query.char_to_ids:
            # Character not in IDS database, return as single token if available
            if char in self.token_to_id:
                return [self.token_to_id[char]]
            else:
                logger.debug(f"Rare character '{char}' not found in vocabulary, using <UNK>")
                return [self.special_tokens['<UNK>']]
        
        try:
            if use_recursive:
                # Use recursive decomposition (better for rare/complex characters)
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
                    logger.debug(f"Token '{token}' not in vocabulary, using <UNK>")
                    token_ids.append(self.special_tokens['<UNK>'])
                    
            return token_ids if token_ids else [self.special_tokens['<UNK>']]
            
        except Exception as e:
            logger.error(f"Error encoding character '{char}': {e}")
            return [self.special_tokens['<UNK>']]
    
    def encode_text(self, text: str, max_length: Optional[int] = None, 
                   add_special_tokens: bool = True, use_recursive: bool = False,
                   enhance_rare_chars: bool = False) -> Dict:
        """
        Encode text to IDS token sequence with rare character enhancement
        
        Args:
            text: Input text string
            max_length: Maximum sequence length (truncate if exceeded)
            add_special_tokens: Whether to add BOS/EOS tokens
            use_recursive: Whether to use recursive decomposition
            enhance_rare_chars: Whether to use enhanced encoding for rare characters
            
        Returns:
            Dictionary containing:
            - input_ids: token IDs
            - attention_mask: attention mask (1 for real tokens, 0 for padding)
            - token_type_ids: token type (0 for structure, 1 for component)
        """
        if not text:
            # Handle empty text
            if add_special_tokens:
                input_ids = [self.special_tokens['<BOS>'], self.special_tokens['<EOS>']]
                token_types = [0, 0]
            else:
                input_ids = [self.special_tokens['<UNK>']]
                token_types = [1]
        else:
            all_token_ids = []
            all_token_types = []  # 0 for IDC, 1 for component
            
            if add_special_tokens:
                all_token_ids.append(self.special_tokens['<BOS>'])
                all_token_types.append(0)
                
            # Process each character with rare character enhancement
            chinese_chars = [char for char in text if '\u4e00' <= char <= '\u9fff']
            
            for i, char in enumerate(chinese_chars):
                # Enhanced encoding for rare characters
                char_tokens = self.encode_char(
                    char, 
                    use_recursive or enhance_rare_chars,
                    prefer_complex_decomposition=enhance_rare_chars
                )
                all_token_ids.extend(char_tokens)
                
                # Determine token types
                for token_id in char_tokens:
                    if token_id in self.id_to_token:
                        token = self.id_to_token[token_id]
                        if token in self.idc_chars:
                            all_token_types.append(0)  # Structure
                        else:
                            all_token_types.append(1)  # Component
                    else:
                        all_token_types.append(1)  # Default to component
                        
                # Add separator between characters (except last)
                if i < len(chinese_chars) - 1:
                    all_token_ids.append(self.special_tokens['<SEP>'])
                    all_token_types.append(0)
                    
            if add_special_tokens:
                all_token_ids.append(self.special_tokens['<EOS>'])
                all_token_types.append(0)
                
            input_ids = all_token_ids
            token_types = all_token_types
            
        # Truncate if necessary
        if max_length and len(input_ids) > max_length:
            input_ids = input_ids[:max_length-1] + [self.special_tokens['<EOS>']]
            token_types = token_types[:max_length-1] + [0]
            
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad if max_length specified
        if max_length:
            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                input_ids.extend([self.special_tokens['<PAD>']] * padding_length)
                token_types.extend([0] * padding_length)
                attention_mask.extend([0] * padding_length)
                
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_types
        }
    
    def encode_text_batch(self, texts: List[str], max_length: Optional[int] = None,
                         add_special_tokens: bool = True, use_recursive: bool = False,
                         enhance_rare_chars: bool = True) -> Dict:
        """
        Encode batch of texts efficiently with rare character support
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length  
            add_special_tokens: Whether to add BOS/EOS tokens
            use_recursive: Whether to use recursive decomposition
            enhance_rare_chars: Whether to use enhanced encoding for rare characters
            
        Returns:
            Dictionary with batched encodings
        """
        batch_encodings = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': []
        }
        
        for text in texts:
            encoding = self.encode_text(
                text, max_length, add_special_tokens, use_recursive, enhance_rare_chars
            )
            batch_encodings['input_ids'].append(encoding['input_ids'])
            batch_encodings['attention_mask'].append(encoding['attention_mask'])
            batch_encodings['token_type_ids'].append(encoding['token_type_ids'])
            
        return batch_encodings
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text with better error handling
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        tokens = []
        special_token_values = set(self.special_tokens.values())
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token_id in special_token_values:
                    continue
                tokens.append(token)
            else:
                logger.warning(f"Unknown token ID: {token_id}")
                if not skip_special_tokens:
                    tokens.append('<UNK>')
                
        return ''.join(tokens)
    
    def get_vocab_info(self) -> Dict:
        """Get comprehensive vocabulary statistics with rare character analysis"""
        idc_count = sum(1 for token in self.token_to_id.keys() if token in self.idc_chars)
        component_count = self.vocab_size - len(self.special_tokens) - idc_count
        
        return {
            'total_vocab_size': self.vocab_size,
            'special_tokens': len(self.special_tokens),
            'idc_chars': idc_count,
            'components': component_count,
            'special_token_names': list(self.special_tokens.keys()),
            'idc_chars_list': sorted(list(self.idc_chars)),
            'preserve_rare_chars': self.preserve_rare_chars,
            'rare_char_stats': self.rare_char_stats,
            'vocab_coverage': self._calculate_coverage()
        }
    
    def _calculate_coverage(self) -> Dict:
        """Calculate vocabulary coverage statistics"""
        total_chars = len(self.ids_query.char_to_ids)
        covered_chars = sum(1 for char in self.ids_query.char_to_ids.keys() 
                          if char in self.token_to_id)
        
        return {
            'total_chars_in_ids_db': total_chars,
            'covered_chars': covered_chars,
            'coverage_ratio': covered_chars / total_chars if total_chars > 0 else 0.0,
            'rare_char_priority': self.preserve_rare_chars
        }
    
    def analyze_rare_characters(self, text: str) -> Dict:
        """
        Analyze rare character coverage and complexity in given text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with rare character analysis
        """
        chinese_chars = [char for char in text if '\u4e00' <= char <= '\u9fff']
        
        if not chinese_chars:
            return {
                'total_chars': 0,
                'rare_chars': [],
                'complex_chars': [],
                'ids_complexity_score': 0.0
            }
        
        rare_chars = []
        complex_chars = []
        total_complexity = 0
        
        for char in chinese_chars:
            # Check if character has IDS decomposition
            if char in self.ids_query.char_to_ids:
                ids_list = self.ids_query.char_to_ids[char]
                
                # Consider rare if has multiple decompositions or complex structure
                if len(ids_list) > 1:
                    rare_chars.append(char)
                
                # Calculate complexity based on IDS length
                max_complexity = max(len(ids) for ids in ids_list) if ids_list else 1
                if max_complexity > 5:  # Arbitrarily complex threshold
                    complex_chars.append(char)
                
                total_complexity += max_complexity
            else:
                # Not in IDS database - potentially very rare
                rare_chars.append(char)
                total_complexity += 1
        
        return {
            'total_chars': len(chinese_chars),
            'rare_chars': rare_chars,
            'complex_chars': complex_chars,
            'rare_char_ratio': len(rare_chars) / len(chinese_chars),
            'complex_char_ratio': len(complex_chars) / len(chinese_chars),
            'ids_complexity_score': total_complexity / len(chinese_chars)
        }
    
    def validate_encoding(self, text: str, encoding: Dict) -> bool:
        """
        Validate that encoding is correct and consistent
        
        Args:
            text: Original text
            encoding: Encoding result from encode_text
            
        Returns:
            True if encoding is valid
        """
        try:
            # Check required keys
            required_keys = ['input_ids', 'attention_mask', 'token_type_ids']
            if not all(key in encoding for key in required_keys):
                return False
                
            # Check length consistency
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            token_type_ids = encoding['token_type_ids']
            
            if not (len(input_ids) == len(attention_mask) == len(token_type_ids)):
                return False
                
            # Check token ID validity
            for token_id in input_ids:
                if token_id < 0 or token_id >= self.vocab_size:
                    return False
                    
            # Check attention mask values
            if not all(mask in [0, 1] for mask in attention_mask):
                return False
                
            # Check token type values
            if not all(ttype in [0, 1] for ttype in token_type_ids):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating encoding: {e}")
            return False