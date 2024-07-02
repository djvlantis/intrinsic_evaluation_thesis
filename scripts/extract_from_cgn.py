import os
import gzip
import re
import nltk

# Download NLTK resources for sentence tokenization
nltk.download('punkt')

def extract_transcriptions_from_file(file_path):
    with gzip.open(file_path, 'rt', encoding='latin-1') as file:
        lines = file.readlines()

    transcriptions = []
    transcription_pattern = re.compile(r'^"(.*?)"$')

    for line in lines:
        match = transcription_pattern.match(line.strip())
        if match:
            transcription = match.group(1)
            if transcription:  # Ignore empty transcriptions
                transcriptions.append(transcription)

    return transcriptions

def clean_text_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        text = input_file.read()

    sentences = nltk.sent_tokenize(text)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for sentence in sentences:
            if sentence.strip():  # Ignore empty lines
                output_file.write(sentence.strip() + '\n')

def extract_transcriptions_from_ort(root_folder_path, output_file_path):
    temp_output_file_path = 'temp_transcriptions.txt'
    with open(temp_output_file_path, 'w', encoding='utf-8') as temp_output_file:
        for folder_name in os.listdir(root_folder_path):
            folder_path = os.path.join(root_folder_path, folder_name)
            if os.path.isdir(folder_path):
                for subfolder_name in os.listdir(folder_path):
                    subfolder_path = os.path.join(folder_path, subfolder_name)
                    if os.path.isdir(subfolder_path):
                        for root, dirs, files in os.walk(subfolder_path):
                            for file in files:
                                if file.endswith('.gz'):
                                    file_path = os.path.join(root, file)
                                    transcriptions = extract_transcriptions_from_file(file_path)
                                    for transcription in transcriptions:
                                        temp_output_file.write(transcription + '\n')
    clean_text_file(temp_output_file_path, output_file_path)
    os.remove(temp_output_file_path)

# Example usage
root_folder_path = 'cgn/CGN_2.0.3/data/annot/text/ort/'
output_file_path = 'training_data/CGN/full_text.txt'
extract_transcriptions_from_ort(root_folder_path, output_file_path)