import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModelForMaskedLM, BertForMaskedLM
import torch
from scipy.stats import spearmanr



parser = argparse.ArgumentParser(description='Run evaluation')
parser.add_argument('--model', type=int, help='model choice: 1 = MBert, 2= RobBERT, 3 = bertje, 4 = XLM RoBERTa, 5 = finetuned_model (path may need adjusting in script)', required=True)
parser.add_argument('--eval_data', default='evaluation_data/SimLex-999-Dutch-final.txt', help='path to evaluation data')

args = parser.parse_args()


if args.model == 1:
    model_name = 'MBert'
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertModel.from_pretrained("bert-base-multilingual-uncased")
elif args.model == 2:
    model_name = 'bertje'
    tokenizer = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
    model = BertModel.from_pretrained('GroNLP/bert-base-dutch-cased')
elif args.model == 3:
    model_name = 'robBERT'
    tokenizer = RobertaTokenizer.from_pretrained('pdelobelle/robbert-v2-dutch-base')
    model = RobertaModel.from_pretrained('pdelobelle/robbert-v2-dutch-base')
elif args.model == 4:
    model_name = 'XLM-Roberta'
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    model = AutoModelForMaskedLM.from_pretrained('xlm-roberta-base')
elif args.model == 5: #replace filepaths as needed
    model_name = 'CGN_finetuned_model_subset_10000'
    model_path = 'finetuned_models/finetuned_model_cgn_sample_10000/pytorch_model.bin'
    tokenizer_path = 'finetuned_models/finetuned_model_cgn_sample_10000'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = BertForMaskedLM.from_pretrained('bert-base-multilingual-uncased', state_dict=torch.load(model_path))
elif args.model == 6: #replace filepaths as needed
    model_name = 'CGN_XLM_Roberta_finetuned_model_subset_10000'
    model_path = 'finetuned_models/finetuned_roberta_model_cgn_sample_10000/pytorch_model.bin'
    tokenizer_path = 'finetuned_models/finetuned_roberta_model_cgn_sample_10000'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForMaskedLM.from_pretrained('xlm-roberta-base', state_dict=torch.load(model_path))


print('Running evaluation for model:', model_name)


def get_word_embedding(word, layer_nums):
    #print("Word:", word)
    subtokens = [tokenizer.cls_token] + tokenizer.tokenize(word) + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(subtokens)

    # Add debugging print statements
    #print("Subtokens:", subtokens)
    #print("Input IDs:", input_ids)

    input_ids = torch.tensor(input_ids).unsqueeze(0)
    # Make sure the model does not compute gradients
    with torch.no_grad():
        # Get the model outputs
        outputs = model(input_ids, output_hidden_states=True)
    # Check if layer_nums is a list or a single integer
    if isinstance(layer_nums, int):
        layer_nums = [layer_nums]
    # Use the hidden state from the specified layers as word embedding
    embeddings = [outputs.hidden_states[i] for i in layer_nums]
    # Average the embeddings from the specified layers
    averaged_embedding = torch.mean(torch.stack(embeddings), dim=0)
    # Ignore the first and the last token ([CLS] and [SEP])
    averaged_embedding = averaged_embedding[:, 1:-1]
    # Get the mean of the subtoken vectors to get the word vector
    word_embedding = torch.mean(averaged_embedding, dim=1)
    # Convert tensor to a numpy array
    word_embedding = word_embedding.numpy()
    return word_embedding


def calculate_similarity(word1, word2, layer_nums):
    word1_embedding = get_word_embedding(word1, layer_nums)
    word2_embedding = get_word_embedding(word2, layer_nums)
    similarity = cosine_similarity(word1_embedding, word2_embedding)[0][0] # removed 1 -
    return similarity


print("Loading evaluation data...")
eval_path = args.eval_data
print(eval_path)
df = pd.read_csv(eval_path, sep='\t')
print(df.head())


print('Evaluation data loaded')

print('Calculating similarity scores for multiple layers...')
# Specify the layers you want to combine
#layer_nums = [i for i in range(13)]
#for layer_num in range(13):
 #   similarity_scores = []  # Initialize similarity_scores in each iteration
  #  for _, row in df.iterrows():
   #     word1 = row['word1']
    #    word2 = row['word2']
     #   similarity = calculate_similarity(word1, word2, layer_nums)
      #  similarity_scores.append(similarity)
    #df[f'predicted_similarity_layer_{layer_num}'] = similarity_scores
    #correlation, _ = spearmanr(df['SimLex999'], df[f'predicted_similarity_layer_{layer_num}'])
    #print(f'Layer {layer_num} - Spearman correlation: {correlation:.3f}')


layer_nums = [i for i in range(13)]
for layer_num in range(13):
    similarity_scores = []  # Initialize similarity_scores in each iteration
    for _, row in df.iterrows():
        word1 = row['word1']
        word2 = row['word2']
        similarity = calculate_similarity(word1, word2, layer_num)  # Use layer_num instead of layer_nums
        similarity_scores.append(similarity)
    df[f'predicted_similarity_layer_{layer_num}'] = similarity_scores
    correlation, _ = spearmanr(df['SimLex999'], df[f'predicted_similarity_layer_{layer_num}'])
    print(f'Layer {layer_num} - Spearman correlation: {correlation:.3f}')

print('Saving results...')
save_name = f'results/{model_name}_manual_0_compounds_.csv'
df.to_csv(save_name, index=False)
print('Results saved to', save_name)
