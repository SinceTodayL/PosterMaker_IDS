"""
Chinese Character IDS (Ideographic Description Sequences) Query Interface
"""

import re
from typing import List, Optional, Dict, Any
from pathlib import Path


class IDSQuery:
    """
    Chinese Character IDS Query Interface
    Provides direct query and recursive decomposition methods
    """
    
    def __init__(self, ids_file_path: str = "ids/ids.txt"):
        """
        Initialize IDS query interface
        
        Args:
            ids_file_path: Path to IDS data file
        """
        self.ids_file_path = Path(ids_file_path)
        self.char_to_ids = {}
        
        # IDS structural descriptors
        self.idc_chars = {
            '⿰', '⿱', '⿲', '⿳', '⿴', '⿵', '⿶', '⿷', '⿸', '⿹', '⿺', '⿻'
        }
        
        self._load_ids_data()
    
    def _load_ids_data(self):
        """Load IDS data from file"""
        # Check if file exists
        if not self.ids_file_path.exists():
            backup_paths = [
                Path("ids/cjkvi-ids/ids.txt"),
                Path("ids/GB2312-ids.txt"),
                Path("./ids/ids.txt")
            ]
            
            for backup_path in backup_paths:
                if backup_path.exists():
                    self.ids_file_path = backup_path
                    break
            else:
                raise FileNotFoundError(f"IDS data file not found: {self.ids_file_path}")
        
        try:
            encodings = ['utf-8', 'utf-8-sig', 'gb2312', 'gbk']
            
            for encoding in encodings:
                try:
                    with open(self.ids_file_path, 'r', encoding=encoding) as f:
                        for line in f:
                            line = line.strip()
                            
                            if not line or line.startswith('#'):
                                continue
                            
                            parts = line.split('\t')
                            if len(parts) < 3:
                                parts = line.split(' ', 2)
                                if len(parts) < 3:
                                    continue
                            
                            try:
                                char = parts[1].strip()
                                if not char or len(char) != 1:
                                    continue
                                
                                ids_list = []
                                for i in range(2, len(parts)):
                                    ids_raw = parts[i].strip()
                                    if ids_raw:
                                        ids_clean = re.sub(r'\[[A-Z]+\]', '', ids_raw).strip()
                                        if ids_clean and ids_clean != char and len(ids_clean) > 0:
                                            if self._is_valid_ids(ids_clean):
                                                ids_list.append(ids_clean)
                                
                                if ids_list:
                                    self.char_to_ids[char] = ids_list
                                    
                            except Exception:
                                continue
                        
                        return
                        
                except UnicodeDecodeError:
                    continue
            
            raise Exception("Failed to decode file with any encoding")
        
        except Exception as e:
            raise Exception(f"Error loading IDS data: {e}")
    
    def _is_valid_ids(self, ids: str) -> bool:
        """Check if IDS sequence is valid"""
        if not ids:
            return False
        
        has_idc = any(char in self.idc_chars for char in ids)
        
        if has_idc and len(ids) > 1:
            return True
        
        if not has_idc and len(ids) == 1:
            return True
        
        return False
    
    def query(self, char: str, recursive: bool = True) -> Optional[Any]:
        """
        Query IDS representation for a character
        
        Args:
            char: Character to query
            recursive: If True, perform recursive decomposition; if False, direct query
            
        Returns:
            IDS representation (list for direct query, dict for recursive query)
            Returns None if character not found
        """
        if recursive:
            return self._query_recursive(char)
        else:
            return self._query_direct(char)
    
    def _query_direct(self, char: str) -> Optional[List[str]]:
        """
        Direct query for character's IDS representation
        
        Args:
            char: Character to query
            
        Returns:
            List of IDS representations, None if not found
        """
        return self.char_to_ids.get(char)
    
    def _query_recursive(self, char: str, depth: int = 0, max_depth: int = 10) -> Dict[str, Any]:
        """
        Recursive decomposition of character structure
        
        Args:
            char: Character to decompose
            depth: Current recursion depth
            max_depth: Maximum recursion depth
            
        Returns:
            Dictionary containing decomposition structure
        """
        if depth > max_depth:
            return {
                'char': char,
                'ids': None,
                'is_basic': True,
                'components': []
            }
        
        ids_list = self._query_direct(char)
        
        if not ids_list:
            return {
                'char': char,
                'ids': None,
                'is_basic': True,
                'components': []
            }
        
        # Use first IDS representation
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
            'components': decomposed_components
        }
    
    def _parse_ids_structure(self, ids: str) -> List[str]:
        """
        Parse IDS structure to extract components
        
        Args:
            ids: IDS sequence
            
        Returns:
            List of components
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
            List of basic components
        """
        recursive_result = self._query_recursive(char)
        components = []
        
        def extract_basic_components(node):
            if node.get('is_basic', False) and not node.get('is_idc', False):
                components.append(node['char'])
            else:
                for component in node.get('components', []):
                    extract_basic_components(component)
        
        extract_basic_components(recursive_result)
        return components
    
    def _flatten_recursive_result(self, result: Dict[str, Any]) -> str:
        """
        Flatten recursive result into a single IDS sequence
        
        Args:
            result: Recursive decomposition result
            
        Returns:
            Flattened IDS sequence string
        """
        if result.get('is_basic', False):
            if result.get('is_idc', False):
                return result['char']
            else:
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
        
        return results 