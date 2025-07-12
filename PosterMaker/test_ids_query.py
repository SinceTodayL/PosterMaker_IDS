#!/usr/bin/env python3
"""
IDS Query Test Interface
"""

from utils.ids_query import IDSQuery


def main():
    """Main test function"""
    ids_query = IDSQuery()
    
    while True:
        user_input = input().strip()
        
        if user_input.lower() in ['quit', 'q']:
            break
        
        if user_input:
            result = ids_query.query_text(user_input)
            print(result)


if __name__ == "__main__":
    main() 