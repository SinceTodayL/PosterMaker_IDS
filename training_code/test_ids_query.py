"""
IDS Query Test Interface
"""

from utils.ids_query import IDSQuery
from utils.ids_tokenizer import IDSTokenizer

def test_ids_query():
    ids_query = IDSQuery()
    
    while True:
        user_input = input().strip()
        
        if user_input.lower() in ['quit', 'q']:
            break
        
        if user_input:
            result = ids_query.query_text(user_input)
            print(result)

def test_ids_tokenizer():
    ids_tokenizer = IDSTokenizer()
    
    while True:
        user_input = input().strip()

        if user_input.lower() in ['quit', 'q']:
            break

        if user_input:
            encoding = ids_tokenizer.encode_text(user_input, max_length=50, add_special_tokens=True)
            print("Input IDs:", encoding['input_ids'])
            print("Attention Mask:", encoding['attention_mask'])
            print("Token Types:", encoding['token_type_ids'])

            print("Decoded:", ids_tokenizer.decode(encoding['input_ids']))


if __name__ == "__main__":
    test_ids_query()