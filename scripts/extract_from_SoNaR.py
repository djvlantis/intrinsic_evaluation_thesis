import os
import random
import xml.etree.ElementTree as ET

# Path to the basedata folder
basedata_folder = "SoNaRCorpus_NC_1.2/SONAR1/COREF/SONAR_1_COREF/Basedata"

save_directory = "training_data"

# Function to extract text from XML file
# Function to extract text from XML file
def extract_text_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    words = [word.text for word in root.findall("word")]
    # Combine words into sentences
    sentences = []
    sentence = []
    for word in words:
        if word.endswith((".", "?", "!")):
            sentence.append(word)
            sentences.append(" ".join(sentence))
            sentence = []
        else:
            sentence.append(word)
    # Combine incomplete sentences
    if sentence:
        if sentences:
            sentences[-1] += " ".join(sentence)
        else:
            sentences.append(" ".join(sentence))
    return sentences

# Function to divide text into subsets of sentences
def divide_text_into_subsets(text, subset_size):
    subsets = []
    current_subset = []
    current_length = 0
    for sentence in text:
        current_subset.append(sentence)
        current_length += len(sentence.split())
        if current_length >= subset_size:
            subsets.append(current_subset)
            current_subset = []
            current_length = 0
    # Add remaining sentences to the last subset
    if current_subset:
        subsets[-1].extend(current_subset)
    return subsets


def assign_text_to_subsets(text_files, subset_sizes, save_directory):
    subsets_mapping = {size: [] for size in subset_sizes}
    remaining_files = text_files.copy()
    
    for size in subset_sizes:
        current_subset = []
        current_length = 0
        
        while current_length < size:
            # Randomly select an XML file (with replacement)
            file = random.choice(remaining_files)
            text = extract_text_from_xml(file)
            file_length = sum(len(sentence.split()) for sentence in text)
            
            # If adding the file doesn't exceed the subset size, add it to the subset
            if current_length + file_length <= size:
                current_subset.extend(text)
                current_length += file_length
                remaining_files.remove(file)
            else:
                break
        
        # Save subset to a text file
        subset_text = "\n".join(current_subset)
        #subset_directory = os.path.join(save_directory, f"subset_{size}")
        #os.makedirs(subset_directory, exist_ok=True)
        #with open(os.path.join(subset_directory, "subset.txt"), "w") as f:
        #    f.write(subset_text)

        with open(save_directory + f"/subset_{size}.txt", "w") as f:
            f.write(subset_text)
        
        subsets_mapping[size].append(current_subset)
    
    return subsets_mapping

# List to store paths of XML files
xml_files = []

# Iterate over XML files in the basedata folder
for filename in os.listdir(basedata_folder):
    if filename.endswith(".xml"):
        xml_files.append(os.path.join(basedata_folder, filename))

# Define sizes of subsets in number of words
subset_sizes = [10000, 50000, 100000, 500000]
subset_sizes = [1000000]

# Randomly assign text from documents to subsets
subsets_mapping = assign_text_to_subsets(xml_files, subset_sizes, save_directory)

# Print the number of documents assigned to each subset size
for size, subsets in subsets_mapping.items():
    print(f"Number of documents assigned to {size}-word subset: {len(subsets)}")


for size, subsets in subsets_mapping.items():
    total_words = sum(len(sentence.split()) for subset in subsets for sentence in subset)
    print(f"Number of words in {size}-word subset: {total_words}")

total_xml_files = len(xml_files)
print(f"Total number of XML files in the basedata folder: {total_xml_files}")

print("Subsets saved successfully.")