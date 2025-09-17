"""

"""

import json
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import torch
from .ids_query import IDSQuery
from .vocab_builder import IDSVocabBuilder, build_ids_vocabulary

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IDSTokenizer:
    """
    Tokenizer for IDS
    Converts IDS sequences to token arrays

    """
    
    def __init__(self, ids_database_path: str, vocab_file: Optional[str] = None, 
                 preserve_rare_chars: bool = True):

        self.ids_database_path = ids_database_path
        self.vocab_file = vocab_file
        self.preserve_rare_chars = preserve_rare_chars

        self.ids_query = IDSQuery(ids_database_path)
        
        self.special_tokens = {}
        
        self.idc_chars = set()
        
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0
        
        self.rare_char_stats = {}
        
        self._initialize_vocabulary()

    def _initialize_vocabulary(self):
        """
        Initialize vocabulary by loading existing file or building new one
        """
        if self.vocab_file and Path(self.vocab_file).exists():
            try:
                self._load_vocab(self.vocab_file)
                logger.info(f"Successfully loaded vocabulary from {self.vocab_file}")
                return
            except Exception as e:
                logger.warning(f"Failed to load vocabulary from {self.vocab_file}: {e}")
                logger.info("Will build new vocabulary...")
        
        # Build vocabulary using standalone builder
        if self.vocab_file:
            # logger.info(f"Building vocabulary and saving to {self.vocab_file}")
            success = build_ids_vocabulary(
                self.ids_database_path, 
                self.vocab_file, 
                self.preserve_rare_chars
            )
            if success:
                self._load_vocab(self.vocab_file)
            else:
                raise RuntimeError("Failed to build vocabulary")
        else:
            # Build vocabulary in memory without saving
            logger.info("Building vocabulary in memory (no save file specified)")
            builder = IDSVocabBuilder(self.ids_database_path)
            vocab_data = builder.build_vocabulary(self.preserve_rare_chars)
            self._load_vocab_from_data(vocab_data)

    def _load_vocab(self, vocab_file: str):
        """
        Load vocabulary from file with enhanced error handling
        
        Args:
            vocab_file: Path to vocabulary file
        """
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self._load_vocab_from_data(vocab_data)
        logger.info(f"Loaded vocabulary with {self.vocab_size} tokens from {vocab_file}")

    def _load_vocab_from_data(self, vocab_data: Dict):
        """
        Load vocabulary from data dictionary
        
        Args:
            vocab_data: Vocabulary data dictionary
        """
        self.token_to_id = vocab_data['token_to_id']
        # Handle both string and int keys in id_to_token mapping
        if 'id_to_token' in vocab_data:
            self.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
        else:
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
            
        self.vocab_size = vocab_data['vocab_size']
        self.special_tokens = vocab_data['special_tokens']
        self.idc_chars = set(vocab_data['idc_chars'])
        self.preserve_rare_chars = vocab_data.get('preserve_rare_chars', True)
        self.rare_char_stats = vocab_data.get('rare_char_stats', {})
        
        # Validate loaded vocabulary
        if len(self.token_to_id) != self.vocab_size:
            raise ValueError(f"Vocabulary size mismatch: {len(self.token_to_id)} != {self.vocab_size}")
        if len(self.id_to_token) != self.vocab_size:
            raise ValueError(f"Reverse mapping size mismatch: {len(self.id_to_token)} != {self.vocab_size}")
            
        if self.rare_char_stats:
            logger.info(f"  Rare character support: {self.rare_char_stats.get('singleton_components', 0)} rare chars")

    def encode_char(self, char: str, use_recursive: bool = False, 
                   prefer_complex_decomposition: bool = False) -> List[int]:
        """
        Encode a single character to IDS token sequence
        Default behavior uses simple decomposition
        
        Args:
            char: Chinese character to encode
            use_recursive: Whether to use recursive decomposition (default: False for training)
            prefer_complex_decomposition: Whether to prefer complex IDS for rare characters (default: False)
            
        Returns:
            List of token IDs representing the character's IDS decomposition
        """
 
        if not char or len(char) != 1:
            return [self.special_tokens['<UNK>']]
        
        '''
        # For training, we default to simple decomposition unless explicitly requested
        # Complex decomposition can be enabled for specific use cases
        if prefer_complex_decomposition and char in self.ids_query.char_to_ids:
            ids_list = self.ids_query.char_to_ids[char]
            # Use recursive if character has complex structure (multiple IDS representations)
            if len(ids_list) > 1 or (ids_list and len(ids_list[0]) > 3):
                use_recursive = True
        '''       

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
        Encode text to IDS token sequence using simple decomposition by default
        
        Args:
            text: Input text string
            max_length: Maximum sequence length (truncate if exceeded)
            add_special_tokens: Whether to add BOS/EOS tokens
            use_recursive: Whether to use recursive decomposition (default: False for training)
            enhance_rare_chars: Whether to use enhanced encoding for rare characters (default: False)
            
        Returns:
            Dictionary containing:
            - input_ids: List of token IDs representing the text
            - attention_mask: Attention mask (1 for real tokens, 0 for padding)
            - token_type_ids: Token type (0 for structure, 1 for component)
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
                
            # Process each character using simple decomposition for training efficiency
            chinese_chars = [char for char in text if '\u4e00' <= char <= '\u9fff']
            
            for i, char in enumerate(chinese_chars):
                # Use simple decomposition by default for training
                char_tokens = self.encode_char(
                    char,
                    use_recursive=use_recursive,
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
                         enhance_rare_chars: bool = False) -> Dict:
        """
        Encode batch of texts efficiently using simple decomposition by default
        
        Args:
            texts: List of text strings to encode
            max_length: Maximum sequence length for each text
            add_special_tokens: Whether to add BOS/EOS tokens
            use_recursive: Whether to use recursive decomposition (default: False for training)
            enhance_rare_chars: Whether to use enhanced encoding for rare characters (default: False)
            
        Returns:
            Dictionary with batched encodings containing input_ids, attention_mask, token_type_ids
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
