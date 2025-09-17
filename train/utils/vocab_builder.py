"""
IDS Vocabulary Builder
"""

import json
import logging
from typing import Dict, Set, Optional
from pathlib import Path
from .ids_query import IDSQuery

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IDSVocabBuilder:

    def __init__(self, ids_database_path: str):

        self.ids_database_path = Path(ids_database_path)
        self.ids_query = IDSQuery(ids_database_path)
        
        self.special_tokens = {
            '<PAD>': 0,
            '<BOS>': 1, 
            '<EOS>': 2,
            '<UNK>': 3,
            '<SEP>': 4,  # Separator between characters
        }
        
        self.idc_chars = {
            '⿰', '⿱', '⿲', '⿳', '⿴', '⿵', '⿶', '⿷', '⿸', '⿹', '⿺', '⿻'
        }
        
        self.rare_char_stats = {
            'total_components': 0,
            'singleton_components': 0,  # Components appearing only once
            'rare_char_coverage': 0.0
        }

    def build_vocabulary(self, preserve_rare_chars: bool = True) -> Dict:
        
        token_to_id = self.special_tokens.copy()
        
        current_id = len(self.special_tokens)
        for idc in sorted(self.idc_chars):
            token_to_id[idc] = current_id
            current_id += 1
            
        component_freq = {}
        
        for char, ids_list in self.ids_query.char_to_ids.items():
            component_freq[char] = component_freq.get(char, 0) + 1
            
            for ids in ids_list:
                for component in ids:
                    if component not in self.idc_chars: 
                        component_freq[component] = component_freq.get(component, 0) + 1
        
        all_components = set(component_freq.keys()) - set(token_to_id.keys())
        
        # Apply character ordering strategy based on preserve_rare_chars setting
        if preserve_rare_chars:
            # Sort by frequency (ascending): rare characters get lower token IDs for better embeddings
            sorted_components = sorted(all_components, key=lambda x: (component_freq.get(x, 0), x))
            logger.info("Vocabulary uses rare-character-first ordering (lower token IDs for rare chars)")
        else:
            # Standard alphabetical sorting for consistent ordering
            sorted_components = sorted(all_components)
            logger.info("Vocabulary uses alphabetical ordering")
        
        for component in sorted_components:
            token_to_id[component] = current_id
            current_id += 1
                
        id_to_token = {v: k for k, v in token_to_id.items()}
        vocab_size = len(token_to_id)
        
        self.rare_char_stats['total_components'] = len(component_freq)
        self.rare_char_stats['singleton_components'] = sum(1 for freq in component_freq.values() if freq == 1)
        self.rare_char_stats['rare_char_coverage'] = (
            self.rare_char_stats['singleton_components'] / self.rare_char_stats['total_components']
            if self.rare_char_stats['total_components'] > 0 else 0.0
        )
        
        '''
        logger.info(f"Built comprehensive vocabulary with {vocab_size} tokens:")
        logger.info(f"  Special tokens: {len(self.special_tokens)}")
        logger.info(f"  IDC chars: {len(self.idc_chars)}")
        logger.info(f"  Total components: {len(all_components)} (ALL preserved)")
        logger.info(f"  Rare characters (freq=1): {self.rare_char_stats['singleton_components']} ({self.rare_char_stats['rare_char_coverage']:.1%})")
        logger.info(f"  Coverage: Enhanced support for rare/complex characters")
        '''
        # Return vocabulary data structure
        vocab_data = {
            'token_to_id': token_to_id,
            'id_to_token': {str(k): v for k, v in id_to_token.items()},  # JSON requires string keys
            'special_tokens': self.special_tokens,
            'idc_chars': list(self.idc_chars),
            'vocab_size': vocab_size,
            'preserve_rare_chars': preserve_rare_chars,
            'rare_char_stats': self.rare_char_stats
        }
        
        return vocab_data

    def save_vocabulary(self, vocab_data: Dict, vocab_file: str) -> None:

        vocab_path = Path(vocab_file)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Vocabulary with rare character support saved to {vocab_file}")

    def build_and_save_vocabulary(self, vocab_file: str, preserve_rare_chars: bool = True) -> Dict:
        
        vocab_data = self.build_vocabulary(preserve_rare_chars)
        self.save_vocabulary(vocab_data, vocab_file)
        return vocab_data

    def validate_vocabulary_file(self, vocab_file: str) -> bool:
        
        try:
            vocab_path = Path(vocab_file)
            if not vocab_path.exists():
                logger.error(f"Vocabulary file does not exist: {vocab_file}")
                return False
                
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            # Check required keys
            required_keys = ['token_to_id', 'special_tokens', 'idc_chars', 'vocab_size']
            for key in required_keys:
                if key not in vocab_data:
                    logger.error(f"Missing required key in vocabulary file: {key}")
                    return False
            
            # Check vocabulary size consistency
            token_to_id = vocab_data['token_to_id']
            vocab_size = vocab_data['vocab_size']
            
            if len(token_to_id) != vocab_size:
                logger.error(f"Vocabulary size mismatch: {len(token_to_id)} != {vocab_size}")
                return False
            
            # Check special tokens
            special_tokens = vocab_data['special_tokens']
            expected_special_tokens = {'<PAD>', '<BOS>', '<EOS>', '<UNK>', '<SEP>'}
            if set(special_tokens.keys()) != expected_special_tokens:
                logger.error(f"Special tokens mismatch. Expected: {expected_special_tokens}, Got: {set(special_tokens.keys())}")
                return False
            
            logger.info(f"Vocabulary file validation passed: {vocab_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating vocabulary file {vocab_file}: {e}")
            return False

    def get_vocabulary_stats(self, vocab_file: str) -> Optional[Dict]:
       
        try:
            if not self.validate_vocabulary_file(vocab_file):
                return None
                
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            token_to_id = vocab_data['token_to_id']
            special_tokens = vocab_data['special_tokens']
            idc_chars = vocab_data['idc_chars']
            
            # Calculate component count
            idc_count = sum(1 for token in token_to_id.keys() if token in idc_chars)
            component_count = len(token_to_id) - len(special_tokens) - idc_count
            
            stats = {
                'total_vocab_size': vocab_data['vocab_size'],
                'special_tokens': len(special_tokens),
                'idc_chars': idc_count,
                'components': component_count,
                'preserve_rare_chars': vocab_data.get('preserve_rare_chars', False),
                'rare_char_stats': vocab_data.get('rare_char_stats', {}),
                'file_path': vocab_file
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting vocabulary stats from {vocab_file}: {e}")
            return None


def build_ids_vocabulary(ids_database_path: str, vocab_file: str, 
                        preserve_rare_chars: bool = True) -> bool:
    
    try:
        builder = IDSVocabBuilder(ids_database_path)
        vocab_data = builder.build_and_save_vocabulary(vocab_file, preserve_rare_chars)
        logger.info(f"Successfully built vocabulary with {vocab_data['vocab_size']} tokens")
        return True
        
    except Exception as e:
        logger.error(f"Failed to build vocabulary: {e}")
        return False


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Build IDS vocabulary from database")
    parser.add_argument("--ids_database", type=str, default="./ids/ids_database.txt",
                       help="Path to IDS database file (default: ./ids/ids_database.txt)")
    parser.add_argument("--vocab_file", type=str, default="./assets/ids_vocab.json",
                       help="Path to save vocabulary file (default: ./assets/ids_vocab.json)")
    parser.add_argument("--preserve_rare_chars", action="store_true", 
                       help="Prioritize rare characters with lower token IDs (default: True)")
    parser.add_argument("--no_preserve_rare_chars", action="store_true",
                       help="Use alphabetical ordering instead of rare-character-first")
    parser.add_argument("--validate_only", action="store_true",
                       help="Only validate existing vocabulary file")
    parser.add_argument("--stats_only", action="store_true", 
                       help="Only show vocabulary statistics")
    
    args = parser.parse_args()
    
    # Determine preserve_rare_chars setting (default: True)
    if args.no_preserve_rare_chars:
        preserve_rare_chars = False
    else:
        preserve_rare_chars = True  # Default behavior
    
    if args.validate_only:
        builder = IDSVocabBuilder(args.ids_database)
        is_valid = builder.validate_vocabulary_file(args.vocab_file)
        print(f"Vocabulary validation: {'PASSED' if is_valid else 'FAILED'}")
        
    elif args.stats_only:
        builder = IDSVocabBuilder(args.ids_database)
        stats = builder.get_vocabulary_stats(args.vocab_file)
        if stats:
            print("Vocabulary Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print("Failed to get vocabulary statistics")
            
    else:
        print(f"Building vocabulary...")
        print(f"  IDS Database: {args.ids_database}")
        print(f"  Vocabulary File: {args.vocab_file}")
        print(f"  Preserve Rare Characters: {preserve_rare_chars}")
        
        success = build_ids_vocabulary(
            args.ids_database, 
            args.vocab_file, 
            preserve_rare_chars
        )
        if success:
            print(f"Vocabulary built successfully: {args.vocab_file}")
        else:
            print("Failed to build vocabulary")
