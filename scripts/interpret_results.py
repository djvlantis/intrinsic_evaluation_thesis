import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def process_file(file_path, output_dir):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    print(file_path)

    # Calculate Spearman correlations
    correlations = []
    for i in range(1, 13):
        correlations.append(df['SimLex999'].corr(df['predicted_similarity_layer_' + str(i)]))

    mean_correlation = sum(correlations) / len(correlations)

    
    # Generate a unique name for the output file based on the CSV filename
    base_filename = os.path.basename(file_path)
    output_filename = os.path.splitext(base_filename)[0] + '_correlation'
    output_filepath = os.path.join(output_dir, output_filename)
    name = base_filename.replace('evaluation_results_', '').replace('.csv', '')
    
    # Plot the correlations in a bar chart
    fig, ax = plt.subplots()
    ax.bar(range(1, 13), correlations)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Spearman Correlation')
    ax.set_ylim(0, 0.5)  # Set y-axis limits
    # ax.axhline(mean_correlation, color='r', linestyle='--', label='Mean Correlation')
    ax.set_title(name)
    
    # Save the plot
    fig.savefig(output_filepath)
    plt.close(fig)  # Close the figure to free up memory


def process_file2(file_path, output_dir):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Calculate Spearman correlations by POS for each layer
    pos_tags = ['N', 'A', 'V']
    correlations_by_pos = {pos: [] for pos in pos_tags}
    
    for i in range(1, 13):
        for pos in pos_tags:
            # Filter the DataFrame for the specific POS
            df_pos = df[df['POS'] == pos]
            if not df_pos.empty:
                corr = df_pos['SimLex999'].corr(df_pos['predicted_similarity_layer_' + str(i)], method='spearman')
            else:
                corr = float('nan')
            correlations_by_pos[pos].append(corr)
    
    # Generate a unique name for the output file based on the CSV filename
    base_filename = os.path.basename(file_path)
    output_filename = os.path.splitext(base_filename)[0] + '_pos_correlation.png'
    output_filepath = os.path.join(output_dir, output_filename)
    name = base_filename.replace('evaluation_results_', '').replace('.csv', '')
    
    # Plot the correlations in a bar chart with three bars per layer
    x = range(1, 13)
    width = 0.2  # Width of the bars
    
    fig, ax = plt.subplots()
    
    ax.bar([p - width for p in x], correlations_by_pos['N'], width=width, label='Nouns (N)', align='center')
    ax.bar(x, correlations_by_pos['A'], width=width, label='Adjectives (A)', align='center')
    ax.bar([p + width for p in x], correlations_by_pos['V'], width=width, label='Verbs (V)', align='center')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Spearman Correlation')
    ax.set_title(name)
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend()
    
    # Save the plot
    fig.savefig(output_filepath)
    plt.close(fig)  # Close the figure to free up memory


def create_correlation_dataframe(csv_files, output_dir):
    data = []

    for file_path in csv_files:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Generate a unique name for the model based on the CSV filename
        base_filename = os.path.basename(file_path)
        model_name = base_filename.replace('evaluation_results_', '').replace('.csv', '')

        # Initialize a dictionary to hold correlations for this model
        model_correlations = {'Model': model_name}

        # Calculate aggregate and POS correlations for each layer
        for i in range(1, 13):
            aggregate_corr = df['SimLex999'].corr(df['predicted_similarity_layer_' + str(i)], method='spearman')
            model_correlations[f'layer{i}_agg'] = aggregate_corr
            
            for pos in ['N', 'A', 'V']:
                df_pos = df[df['POS'] == pos]
                if not df_pos.empty:
                    pos_corr = df_pos['SimLex999'].corr(df_pos['predicted_similarity_layer_' + str(i)], method='spearman')
                else:
                    pos_corr = float('nan')
                model_correlations[f'layer{i}_{pos}'] = pos_corr
        
        # Append the model correlations to the data list
        data.append(model_correlations)

    # Create a DataFrame from the collected data
    correlation_df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    output_filepath = os.path.join(output_dir, 'correlation_summary.csv')
    correlation_df.to_csv(output_filepath, index=False)



def main(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    # Process each CSV file
    for csv_file in csv_files:
        process_file(csv_file, output_dir)

    for csv_file in csv_files:
        process_file2(csv_file, output_dir)
    
    create_correlation_dataframe(csv_files, output_dir)
    


if __name__ == '__main__':
    input_directory = 'results'  # Specify the input directory containing CSV files
    output_directory = 'visualisation'  # Specify the output directory for saving plots

    
    main(input_directory, output_directory)
