"""
Configuration loader for the PosterMaker IDS-based training pipeline.
Provides utilities to load and validate training configurations from YAML files.
"""

import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary loaded from the YAML file
        
    Raises:
        FileNotFoundError: If the configuration file does not exist
        yaml.YAMLError: If the YAML file is malformed
        Exception: For other unexpected errors during loading
    """
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        if config is None:
            raise ValueError(f"Configuration file is empty or invalid: {config_path}")
            
        return config
        
    except FileNotFoundError:
        raise
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration file {config_path}: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error loading configuration from {config_path}: {e}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate the loaded configuration dictionary for required keys and structure.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_keys = [
        'poster_maker_dir',
        'dataset_dir', 
        'output_dir',
        'stage1',
        'project_name'
    ]
    
    # Check for required top-level keys
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required configuration key: {key}")
            return False
    
    # Check for required stage1 hyperparameters
    stage1_required = [
        'learning_rate',
        'batch_size',
        'num_train_epochs',
        'embedding_dim',
        'max_seq_length'
    ]
    
    if 'stage1' in config:
        for key in stage1_required:
            if key not in config['stage1']:
                print(f"Error: Missing required stage1 configuration key: {key}")
                return False
    
    return True


def get_absolute_path(config: Dict[str, Any], relative_path: str, base_key: str = 'poster_maker_dir') -> str:
    """
    Convert a relative path to an absolute path based on a base directory from config.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        relative_path (str): Relative path to convert
        base_key (str): Key in config containing the base directory path
        
    Returns:
        str: Absolute path
    """
    base_dir = config.get(base_key, '.')
    return os.path.abspath(os.path.join(base_dir, relative_path))


def setup_output_directories(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Create and return paths for all output directories needed during training.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        Dict[str, str]: Dictionary mapping directory names to their paths
    """
    output_base = config.get('output_dir', './training_output')
    
    directories = {
        'base': output_base,
        'checkpoints': os.path.join(output_base, 'checkpoints'),
        'logs': os.path.join(output_base, 'logs'),
        'samples': os.path.join(output_base, 'samples'),
        'tensorboard': os.path.join(output_base, 'tensorboard')
    }
    
    # Create directories if they don't exist
    for dir_name, dir_path in directories.items():
        os.makedirs(dir_path, exist_ok=True)
        
    return directories


if __name__ == "__main__":
    # Example usage and testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Test configuration loading")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        print("Configuration loaded successfully:")
        print(yaml.dump(config, default_flow_style=False, indent=2))
        
        if validate_config(config):
            print("Configuration validation passed")
            
            # Setup output directories
            output_dirs = setup_output_directories(config)
            print("Output directories created:")
            for name, path in output_dirs.items():
                print(f"  {name}: {path}")
        else:
            print("Configuration validation failed")
            
    except Exception as e:
        print(f"Error: {e}")