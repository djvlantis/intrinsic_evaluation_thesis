import logging
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModelForMaskedLM

# Suppress warnings from transformers library
logging.getLogger("transformers").setLevel(logging.ERROR)

def tokenize_and_print(text):
    # Define the models and their corresponding tokenizers
    models_and_tokenizers = {
        'MBert': ('bert-base-multilingual-uncased', BertTokenizer, BertModel),
        'bertje': ('GroNLP/bert-base-dutch-cased', BertTokenizer, BertModel),
        'robBERT': ('pdelobelle/robbert-v2-dutch-base', RobertaTokenizer, RobertaModel),
        'XLM-Roberta': ('xlm-roberta-base', AutoTokenizer, AutoModelForMaskedLM)
    }

    # Iterate over each model and tokenize the text
    for model_name, (model_id, tokenizer_class, model_class) in models_and_tokenizers.items():
        tokenizer = tokenizer_class.from_pretrained(model_id)
        model = model_class.from_pretrained(model_id)

        # Tokenize the input text
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Display the tokenized text
        print(f"Model: {model_name}")
        print(f"Original Text: {text}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print("\n" + "="*50 + "\n")

# Example usage
text_to_tokenize = "verzamelen"
tokenize_and_print(text_to_tokenize)
