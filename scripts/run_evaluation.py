import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch


parser = argparse.ArgumentParser(description='Run evaluation')
parser.add_argument('--model', type=int, help='model choice: 1 = MBert, 2= RobBERT, 3 = XLM-RoBERTa')
parser.add_argument('--eval_data', default = 'evaluation_data/SimLex-999-Dutch-final.txt', help='path to evaluation data')
args = parser.parse_args()




if args.model == 1:
    model_name = 'MBert'
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertModel.from_pretrained("bert-base-multilingual-uncased")
elif args.model == 2:
    model = 'robbert'
elif args.model == 3:
    model = 'xlm-roberta'
print('Running evaluation for model: ', model)



def get_embedding(word):
    # Tokenize the word
    tokens = tokenizer.tokenize(word)
    # Convert tokens to token IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Convert token IDs to tensor
    token_ids_tensor = torch.tensor([token_ids])
    # Get embeddings
    with torch.no_grad():
        outputs = model(token_ids_tensor)
    # Extract embeddings for the word
    embedding = outputs.last_hidden_state[0][0]
    return embedding


def get_similarity_score(emb1, emb2):
    emb1 = emb1.reshape(1, -1)
    emb2 = emb2.reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]






print("loading evaluation data...")
eval_path = args.eval_data
df = pd.read_csv(eval_path, sep='\t')
print('Evaluation data loaded')

# load embeddings
print('loading embeddings...')
df['emb1'] = df['word1'].apply(get_embedding)
df['emb2'] = df['word2'].apply(get_embedding)
print('Embeddings loaded')

print(df.head())

print('calculating similarity scores')
df['similarity_score'] = df.apply(lambda row: get_similarity_score(row['emb1'], row['emb2']), axis=1)
print('Similarity scores calculated')

print('Saving results...')
save_name = 'results/evaluation_results_'+model_name+'.csv'
df.to_csv(save_name, index=False)
