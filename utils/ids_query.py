"""
    source: ids/ids_database.txt
    eg. 
    {'橋': 
        {
            'simple': '⿰木喬', 
            'recursive': '⿰⿻⿻一丨𠆢⿱⿱⿱㇒⿻一人口⿵冂口'
        }
    }

"""
import re
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IDSQuery:
    """
    Chinese Character IDS Query Interface
    Provides direct query and recursive decomposition methods
    """
    
    def __init__(self, ids_database_path: str):
        """
        Initialize IDS query interface
        
        Args:
            ids_database_path: Path to IDS database file (ids_database.txt)
        """
        self.ids_database_path = Path(ids_database_path)
        self.char_to_ids = {}
        
        # IDS structural descriptors (12 official IDCs)
        self.idc_chars = {
            '⿰', '⿱', '⿲', '⿳', '⿴', '⿵', '⿶', '⿷', '⿸', '⿹', '⿺', '⿻'
        }
        
        # Statistics for monitoring
        self.stats = {
            'total_chars_loaded': 0,
            'total_ids_sequences': 0
        }
        
        self._load_ids_data()
    
    def _load_ids_data(self):
        """
        Load IDS data from the Unicode IDS database file (ids_database.txt).
        This method is specifically designed to parse the official Unicode format.
        Format per line: U+CODE\tCHAR\t^IDS1$(SOURCE1)\t^IDS2$(SOURCE2)...
        """
        if not self.ids_database_path.exists():
            error_msg = f"IDS data file not found: {self.ids_database_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            chars_processed = 0
            sequences_processed = 0
            
            with open(self.ids_database_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#') or line.startswith(';'):
                        continue
                    
                    # Ensure it's a valid data line
                    if not line.startswith('U+'):
                        continue

                    parts = line.split('\t')
                    if len(parts) < 3:
                        logger.debug(f"Skipping line {line_num}: insufficient parts")
                        continue

                    char = parts[1].strip()
                    if not char or len(char) != 1:
                        logger.debug(f"Skipping line {line_num}: invalid character '{char}'")
                        continue

                    if char not in self.char_to_ids:
                        self.char_to_ids[char] = []
                        chars_processed += 1

                    # Process each IDS sequence for the character
                    char_sequences_added = 0
                    for i in range(2, len(parts)):
                        ids_part = parts[i].strip()
                        
                        # Extract the core IDS sequence, which is between ^ and $
                        match = re.search(r'\^(.+?)\$', ids_part)
                        if match:
                            ids_clean = match.group(1)
                            
                            # Add to list if it's a valid, new decomposition
                            if self._is_valid_ids(ids_clean) and ids_clean not in self.char_to_ids[char]:
                                self.char_to_ids[char].append(ids_clean)
                                char_sequences_added += 1
                                sequences_processed += 1
                    
                    if char_sequences_added == 0:
                        logger.debug(f"No valid IDS sequences found for character '{char}' on line {line_num}")

            # Update statistics
            self.stats['total_chars_loaded'] = chars_processed
            self.stats['total_ids_sequences'] = sequences_processed
            
            
            

        except Exception as e:
            error_msg = f"Error loading IDS data from {self.ids_database_path}: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def _is_valid_ids(self, ids: str) -> bool:
        """
        Check if IDS sequence is valid
        
        Args:
            ids: IDS sequence string
            
        Returns:
            True if valid IDS sequence
        """
        if not ids:
            return False
        
        has_idc = any(char in self.idc_chars for char in ids)
        
        # Valid IDS should either have structure (IDC + components) or be a single character
        if has_idc and len(ids) > 1:
            return True
        
        if not has_idc and len(ids) == 1:
            return True
        
        return False
    
    def query(self, char: str, recursive: bool = False) -> Optional[Any]:
        """
        Query IDS representation for a character
        
        Args:
            char: Character to query
            recursive: If True, perform recursive decomposition; if False, direct query
            
        Returns:
            IDS representation (list for direct query, dict for recursive query)
            Returns None if character not found
        """
        if not char or len(char) != 1:
            logger.warning(f"Invalid character input: '{char}'")
            return None
            
        try:
            if recursive:
                return self._query_recursive(char)
            else:
                return self._query_direct(char)
        except Exception as e:
            logger.error(f"Error querying character '{char}': {e}")
            return None
    
    def _query_direct(self, char: str) -> Optional[List[str]]:
        """
        Direct query for character's IDS representation
        
        Args:
            char: Character to query
            
        Returns:
            List of IDS representations, None if not found
        """
        return self.char_to_ids.get(char)
    
    def _query_recursive(self, char: str, depth: int = 0, max_depth: int = 5) -> Dict[str, Any]:
        """
        Recursive decomposition of character structure
        
        Args:
            char: Character to decompose
            depth: Current recursion depth
            max_depth: Maximum recursion depth to prevent infinite loops
            
        Returns:
            Dictionary containing decomposition structure
        """
        if depth > max_depth:
            return {
                'char': char,
                'ids': None,
                'is_basic': True,
                'components': [],
                'depth_exceeded': True
            }
        
        ids_list = self._query_direct(char)
        
        if not ids_list:
            return {
                'char': char,
                'ids': None,
                'is_basic': True,
                'components': []
            }
        
        # Use first IDS representation (most common/standard)
        ids = ids_list[0]
        
        # Parse IDS structure
        components = self._parse_ids_structure(ids)
        
        # Recursively decompose each component
        decomposed_components = []
        for component in components:
            if component in self.idc_chars:
                # Structural descriptor, no further decomposition needed
                decomposed_components.append({
                    'char': component,
                    'ids': None,
                    'is_basic': True,
                    'is_idc': True,
                    'components': []
                })
            else:
                # Recursively decompose sub-component
                decomposed_components.append(
                    self._query_recursive(component, depth + 1, max_depth)
                )
        
        return {
            'char': char,
            'ids': ids,
            'is_basic': False,
            'components': decomposed_components,
            'depth': depth
        }
    
    def _parse_ids_structure(self, ids: str) -> List[str]:
        """
        Parse IDS structure to extract components
        
        Args:
            ids: IDS sequence
            
        Returns:
            List of components (IDCs and characters)
        """
        components = []
        i = 0
        
        while i < len(ids):
            char = ids[i]
            components.append(char)
            i += 1
        
        return components
    
    def get_basic_components(self, char: str) -> List[str]:
        """
        Get all basic components (leaf nodes) of a character
        
        Args:
            char: Character to analyze
            
        Returns:
            List of basic components (excluding IDCs)
        """
        try:
            recursive_result = self._query_recursive(char)
            components = []
            
            def extract_basic_components(node):
                if node.get('is_basic', False) and not node.get('is_idc', False):
                    components.append(node['char'])
                else:
                    for component in node.get('components', []):
                        extract_basic_components(component)
            
            extract_basic_components(recursive_result)
            return list(set(components))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting components for '{char}': {e}")
            return []
    
    def _flatten_recursive_result(self, result: Dict[str, Any]) -> str:
        """
        Flatten recursive result into a single IDS sequence
        
        Args:
            result: Recursive decomposition result
            
        Returns:
            Flattened IDS sequence string
        """
        if result.get('is_basic', False):
            return result['char']
        
        # For non-basic components, collect all parts
        parts = []
        for component in result.get('components', []):
            parts.append(self._flatten_recursive_result(component))
        
        return ''.join(parts)
    
    def query_text(self, text: str) -> List[Dict[str, Dict[str, str]]]:
        """
        Query IDS for text, returning both direct and recursive sequences
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries, each containing:
            {char: {'simple': direct_ids, 'recursive': recursive_ids}}
        """
        results = []
        
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # Chinese characters only
                try:
                    # Direct query
                    direct_ids = self._query_direct(char)
                    simple_ids = direct_ids[0] if direct_ids else None
                    
                    # Recursive query
                    recursive_result = self._query_recursive(char)
                    if recursive_result and not recursive_result.get('is_basic', False):
                        recursive_ids = self._flatten_recursive_result(recursive_result)
                    else:
                        recursive_ids = char  # Basic character
                    
                    results.append({
                        char: {
                            'simple': simple_ids,
                            'recursive': recursive_ids
                        }
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing character '{char}': {e}")
                    results.append({
                        char: {
                            'simple': None,
                            'recursive': char
                        }
                    })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the loaded IDS data"""
        return {
            **self.stats,
            'idc_chars_count': len(self.idc_chars),
            'chars_with_multiple_ids': sum(
                1 for ids_list in self.char_to_ids.values() if len(ids_list) > 1
            ),
            'max_ids_per_char': max(
                len(ids_list) for ids_list in self.char_to_ids.values()
            ) if self.char_to_ids else 0
        }