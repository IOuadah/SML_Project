import numpy as np
import matplotlib.pyplot as plt

def plot_cluster_counts(*dataframes, exclude_label='CID'):
    """
    Plot the number of samples in each cluster for any number of dataframes.
    
    Parameters:
    - *dataframes: A variable number of pandas DataFrames with one-hot encodings for clusters.
    - exclude_label: A label to exclude from the counts, default is 'CID'.
    """
    # Initializing an empty dictionary to hold the counts for each dataframe
    counts_dict = {}
    
    # Iterating over each dataframe and calculating the cluster counts
    for i, df in enumerate(dataframes):
        counts = df.sum().sort_values(ascending=False)
        if exclude_label in counts:
            counts = counts.drop(exclude_label)
        counts_dict[f'df{i+1}_counts'] = counts.values
    
    # Creating a DataFrame from the dictionary
    combined_df = pd.DataFrame(counts_dict)

    print(combined_df)
    
    # Optional: Calculating a total count across all datasets for ordering
    combined_df['Total'] = combined_df.sum(axis=1)
    combined_df = combined_df.sort_values(by='Total', ascending=False).drop(columns=['Total'])
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(combined_df.shape[0])  # Cluster positions
    width = 0.8 / len(dataframes)  # Width of the bars, adjusted for the number of dataframes
    
    # Generate bars for each dataset
    for i, col in enumerate(combined_df.columns):
        ax.bar(x + i*width, combined_df[col], width, label=col)
    
    # Formatting the plot
    ax.set_ylabel('Counts')
    ax.set_title('Counts by cluster and dataset')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f'{i+1}' for i in x])  # Adjusting labels if necessary
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_cluster_counts_array(array1, array2):
    """
    Calculates and normalizes the column sums of two 2D NumPy arrays, plots these normalized sums side by side in a histogram for comparison of ratios, and annotates the bars with the absolute counts.
    
    Parameters:
    - array1: A 2D NumPy array.
    - array2: A 2D NumPy array.
    """
    # Calculate column sums
    sums_array1 = np.sum(array1, axis=0)
    sums_array2 = np.sum(array2, axis=0)
    
    # Normalize the column sums by the total sum of each array to get ratios
    norm_sums_array1 = sums_array1 / np.sum(sums_array1)
    norm_sums_array2 = sums_array2 / np.sum(sums_array2)
    
    # Setting the positions for the bars
    n_columns = max(len(norm_sums_array1), len(norm_sums_array2))
    bar_width = 0.35  # Width of the bars
    index = np.arange(n_columns)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(index, norm_sums_array1, bar_width, alpha=0.5, label='Array 1 Column Ratios')
    bars2 = plt.bar(index + bar_width, norm_sums_array2, bar_width, alpha=0.5, label='Array 2 Column Ratios')

    # Annotating the bars with the absolute counts
    for bar, label in zip(bars1, sums_array1):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{int(label)}', ha='center', va='bottom')

    for bar, label in zip(bars2, sums_array2):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{int(label)}', ha='center', va='bottom')
    
    plt.xlabel('Column Index')
    plt.ylabel('Normalized Column Sum')
    plt.title('Normalized Histogram of Column Sums Side by Side with Absolute Counts')
    plt.xticks(index + bar_width / 2, labels=index)  # Set x-tick labels to the middle of the group of bars
    plt.legend()
    
    plt.tight_layout()
    plt.show()


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from scipy.stats import randint as sp_randint
import numpy as np

