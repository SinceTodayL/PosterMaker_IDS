"""
Data preparation and validation script for the PosterMaker IDS-based training pipeline.
This script validates the dataset structure and contents to ensure compatibility with the training pipeline.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

from src.config import load_config


def setup_logging(log_level: str = "INFO") -> None:
    """
    Setup logging configuration for the script.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/data_validation.log', mode='w')
        ]
    )


def validate_sample_folder(sample_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate a single sample folder for required files and content.
    
    Args:
        sample_path (Path): Path to the sample folder
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for required files
    # Check for image file (multiple formats supported)
    image_file = None
    for ext in ['png', 'jpg', 'jpeg']:
        candidate = sample_folder / f'image.{ext}'
        if candidate.exists():
            image_file = candidate
            break
    
    required_files = ['subject_mask.png', 'annotation.json']
    for required_file in required_files:
        file_path = sample_path / required_file
        if not file_path.exists():
            errors.append(f"Missing required file: {file_path}")
    
    # Validate annotation.json if it exists
    annotation_path = sample_path / 'annotation.json'
    if annotation_path.exists():
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)
            
            # Check for required keys
            required_keys = ['prompt', 'texts']
            for key in required_keys:
                if key not in annotation_data:
                    errors.append(f"Missing required key '{key}' in annotation.json: {annotation_path}")
                elif not annotation_data[key]:
                    # Check if the value is empty or None
                    if isinstance(annotation_data[key], str) and not annotation_data[key].strip():
                        errors.append(f"Empty value for key '{key}' in annotation.json: {annotation_path}")
                    elif isinstance(annotation_data[key], list) and len(annotation_data[key]) == 0:
                        errors.append(f"Empty list for key '{key}' in annotation.json: {annotation_path}")
                    elif annotation_data[key] is None:
                        errors.append(f"Null value for key '{key}' in annotation.json: {annotation_path}")
            
            # Validate texts field structure
            if 'texts' in annotation_data and isinstance(annotation_data['texts'], list):
                for i, text_item in enumerate(annotation_data['texts']):
                    if not isinstance(text_item, dict):
                        errors.append(f"Text item {i} is not a dictionary in annotation.json: {annotation_path}")
                    elif 'text' not in text_item or not text_item['text']:
                        errors.append(f"Text item {i} missing or empty 'text' field in annotation.json: {annotation_path}")
                        
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON format in annotation.json: {annotation_path} - {str(e)}")
        except Exception as e:
            errors.append(f"Error reading annotation.json: {annotation_path} - {str(e)}")
    
    return len(errors) == 0, errors


def validate_dataset(dataset_dir: str) -> Dict[str, Any]:
    """
    Validate the entire dataset structure and contents.
    
    Args:
        dataset_dir (str): Path to the dataset root directory
        
    Returns:
        Dict[str, Any]: Validation summary report
    """
    logger = logging.getLogger(__name__)
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        logger.error(f"Dataset directory does not exist: {dataset_dir}")
        return {
            'status': 'failed',
            'error': f"Dataset directory does not exist: {dataset_dir}",
            'total_samples': 0,
            'valid_samples': 0,
            'error_count': 1
        }
    
    # Initialize counters
    total_samples = 0
    valid_samples = 0
    all_errors = []
    
    # Check both train and val subdirectories
    for split in ['train', 'val']:
        split_path = dataset_path / split
        
        if not split_path.exists():
            error_msg = f"Missing {split} subdirectory: {split_path}"
            logger.warning(error_msg)
            all_errors.append(error_msg)
            continue
            
        logger.info(f"Validating {split} split...")
        
        # Iterate through sample folders
        sample_folders = [d for d in split_path.iterdir() if d.is_dir()]
        split_total = len(sample_folders)
        split_valid = 0
        
        for sample_folder in sample_folders:
            total_samples += 1
            is_valid, errors = validate_sample_folder(sample_folder)
            
            if is_valid:
                valid_samples += 1
                split_valid += 1
            else:
                for error in errors:
                    logger.error(error)
                    all_errors.append(error)
        
        logger.info(f"Split {split}: {split_valid}/{split_total} valid samples")
    
    # Generate summary report
    error_count = len(all_errors)
    success_rate = (valid_samples / total_samples * 100) if total_samples > 0 else 0
    
    report = {
        'status': 'completed',
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'error_count': error_count,
        'success_rate': success_rate,
        'errors': all_errors[:50]  # Limit to first 50 errors for readability
    }
    
    # Print summary
    logger.info("=" * 60)
    logger.info("DATASET VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples processed: {total_samples}")
    logger.info(f"Valid samples: {valid_samples}")
    logger.info(f"Samples with errors: {total_samples - valid_samples}")
    logger.info(f"Total errors found: {error_count}")
    logger.info(f"Success rate: {success_rate:.2f}%")
    
    if error_count > 0:
        logger.warning(f"Found {error_count} errors in the dataset. Please check the log for details.")
        if error_count > 50:
            logger.warning("Only the first 50 errors are shown in the report.")
    else:
        logger.info("Dataset validation completed successfully with no errors!")
    
    return report


def main():
    """
    Main function to handle command-line arguments and execute dataset validation.
    """
    parser = argparse.ArgumentParser(
        description="Validate PosterMaker dataset structure and contents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python prepare_data.py --config configs/train_config.yaml
    python prepare_data.py --config configs/train_config.yaml --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the training configuration YAML file'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Extract dataset directory from config
        dataset_dir = config.get('dataset_dir')
        if not dataset_dir:
            raise ValueError("'dataset_dir' not found in configuration file")
            
        logger.info(f"Dataset directory: {dataset_dir}")
        
        # Validate dataset
        report = validate_dataset(dataset_dir)
        
        # Save validation report
        report_path = Path('logs') / 'validation_report.json'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Validation report saved to: {report_path}")
        
        # Exit with appropriate code
        if report['error_count'] == 0:
            logger.info("Dataset validation completed successfully!")
            return 0
        else:
            logger.error("Dataset validation completed with errors. Please review the log.")
            return 1
            
    except Exception as e:
        logger.error(f"Failed to validate dataset: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())