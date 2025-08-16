"""
IDS Query Test Interface
"""

from src.utils.ids_query import IDSQuery
from src.utils.ids_tokenizer import IDSTokenizer
from src.models.ids_text_embedder import IDSEmbedding

def test_ids_query():
    ids_query = IDSQuery('ids/ids_database.txt')
    
    while True:
        user_input = input().strip()
        
        if user_input.lower() in ['quit', 'q']:
            break
        
        if user_input:
            result = ids_query.query_text(user_input)
            print(result)

def test_ids_tokenizer():

    vocab_file = 'assets/ids_vocab.json'
    ids_tokenizer = IDSTokenizer(
        ids_database_path='ids/ids_database.txt',
        vocab_file=vocab_file,
        preserve_rare_chars=True
    )
    
    while True:
        user_input = input().strip()

        if user_input.lower() in ['quit', 'q']:
            break

        if user_input:
            encoding = ids_tokenizer.encode_text(user_input, use_recursive=False)
            print("Input IDs:", encoding['input_ids'])
            print("Attention Mask:", encoding['attention_mask'])
            print("Token Types:", encoding['token_type_ids'])

            print("Decoded:", ids_tokenizer.decode(encoding['input_ids']))

def test_ids_embedder():
    ids_embedder = IDSEmbedding(    
        vocab_size=10000,
        embed_dim=64,
        max_seq_length=128,
        dropout=0.1,
        char2feat_path='assets/char2feat.json'
    )

if __name__ == "__main__":
    test_ids_embedder()