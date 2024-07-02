import random

def select_random_lines(input_file_path, output_file_path, num_lines):
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()

    selected_lines = random.sample(lines, min(num_lines, len(lines)))

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in selected_lines:
            output_file.write(line)

# Example usage
input_file_path = 'training_data/CGN/full_text.txt'  # Replace with the path to your input text file
output_file_path = 'training_data/CGN/cgn_sample_10000.txt'  # Replace with the desired path for the output text file
num_lines = 10000 # Number of randomly selected lines

select_random_lines(input_file_path, output_file_path, num_lines)
