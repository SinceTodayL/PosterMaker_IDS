from utils.utils import get_char_features_by_text, get_positional_encoding
import torch

if __name__ == "__main__":
    texts = [
                {"content": "护肤美颜贵妇乳", "pos": [69, 104, 681, 185]},
                {"content": "99.9%纯度玻色因", "pos": [165, 226, 585, 272]},
                {"content": "持久保年轻", "pos": [266, 302, 483, 347]}
            ]
    feature_dict_path='./assets/char2feat_ppocr_neck64_avg.pth'
    feature_dict = torch.load(feature_dict_path)
    char_padding_to_len = 16
    char_pos_encoding_dim = 32
    text_pos_encoding_dim = 32

    after_text_embedder = get_char_features_by_text(texts, feature_dict, char_padding_to_len)
    char_positional_encoding = get_positional_encoding(char_padding_to_len, char_pos_encoding_dim) # N*32

    print("after_text_embedder: ")
    print(after_text_embedder)
    print("shape: ")
    print(after_text_embedder.shape)
    print("char_positional_encoding: ")
    print(char_positional_encoding)

