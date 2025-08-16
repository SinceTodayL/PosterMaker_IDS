import torch
import os


def load_char2feat(char2feat_path):
    """
    Load character feature dictionary from pth file
    
    Args:
        char2feat_path: Path to char2feat_ppocr_neck64_avg.pth file
        
    Returns:
        dict: Character to feature vector mapping
    """
    if not os.path.exists(char2feat_path):
        print(f"File not found: {char2feat_path}")
        return None
    
    try:
        char2feat = torch.load(char2feat_path, map_location='cpu')
        return char2feat
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def analyze_char2feat_structure(char2feat):
    """
    Analyze the structure of char2feat dictionary
    
    Args:
        char2feat: Character feature dictionary
    """
    print("=== Character Feature Analysis ===")
    print(f"Total characters: {len(char2feat)}")
    print(f"Data type: {type(char2feat)}")
    
    # Get sample characters
    sample_chars = list(char2feat.keys())[:10]
    print(f"Sample characters: {sample_chars}")
    
    # Check feature dimensions
    if len(char2feat) > 0:
        first_char = list(char2feat.keys())[0]
        first_feat = char2feat[first_char]
        print(f"Feature shape: {first_feat.shape}")
        print(f"Feature dtype: {first_feat.dtype}")
        print(f"Feature range: [{first_feat.min().item():.4f}, {first_feat.max().item():.4f}]")


def get_char_feature(char2feat, character):
    """
    Get feature vector for a specific character
    
    Args:
        char2feat: Character feature dictionary
        character: Target character
        
    Returns:
        torch.Tensor: Feature vector for the character
    """
    if character in char2feat:
        return char2feat[character]
    else:
        # Use space character as default fallback
        if ' ' in char2feat:
            print(f"Character '{character}' not found, using space as fallback")
            return char2feat[' ']
        else:
            print(f"Character '{character}' not found, no fallback available")
            return None


def get_text_features(char2feat, text, max_length=16):
    """
    Convert text to feature matrix using char2feat
    
    Args:
        char2feat: Character feature dictionary
        text: Input text string
        max_length: Maximum sequence length for padding
        
    Returns:
        torch.Tensor: Feature matrix of shape (max_length, feature_dim)
    """
    # Get feature dimension from first character
    first_char = list(char2feat.keys())[0]
    feature_dim = char2feat[first_char].shape[0]
    
    # Initialize feature matrix
    text_features = torch.zeros(max_length, feature_dim)
    
    # Fill features for each character
    for i, char in enumerate(text):
        if i >= max_length:
            break
            
        char_feat = get_char_feature(char2feat, char)
        if char_feat is not None:
            text_features[i] = char_feat
    
    return text_features


def demonstrate_usage():
    """
    Demonstrate how to use char2feat file
    """
    # Load char2feat file
    char2feat_path = "assets/char2feat_ppocr_neck64_avg.pth"
    char2feat = load_char2feat(char2feat_path)
    
    if char2feat is None:
        print("Failed to load char2feat file")
        return
    
    # Analyze structure
    analyze_char2feat_structure(char2feat)
    
    # Test single character feature extraction
    print("\n=== Single Character Feature ===")
    test_char = "中"
    char_feat = get_char_feature(char2feat, test_char)
    if char_feat is not None:
        print(f"Character '{test_char}' feature shape: {char_feat.shape}")
        print(f"First 5 values: {char_feat[:5].tolist()}")
    
    # Test text feature extraction
    print("\n=== Text Feature Extraction ===")
    test_text = "中国"
    text_feat = get_text_features(char2feat, test_text, max_length=8)
    print(f"Text '{test_text}' feature shape: {text_feat.shape}")
    print(f"Non-zero rows: {(text_feat.sum(dim=1) != 0).sum().item()}")
    
    # Show available characters (first 20)
    print("\n=== Available Characters (first 20) ===")
    available_chars = list(char2feat.keys())[:20]
    print(f"Characters: {available_chars}")
    
    # Check common characters
    print("\n=== Common Character Coverage ===")
    common_chars = ['中', '国', '人', '的', '一', '是', '在', '有', '不', '了']
    for char in common_chars:
        status = "YES" if char in char2feat else "NO"
        print(f"'{char}': {status}")


if __name__ == "__main__":
    demonstrate_usage()