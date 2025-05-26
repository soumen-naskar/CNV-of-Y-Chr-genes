import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_statistical_analysis(data, group_col, value_col):
    groups = [group for _, group in data.groupby(group_col)[value_col]]
    f_stat, p_value = stats.f_oneway(*groups)
    tukey = pairwise_tukeyhsd(data[value_col], data[group_col])
    return f_stat, p_value, tukey

def create_plots(data, group_cols, value_cols, output_file):
    genes = len(value_cols)
    group_types = len(group_cols)
    
    fig, axes = plt.subplots(group_types, genes, figsize=(20, 12))
    plt.subplots_adjust(hspace=0.5)
    
    # Color schemes
    genetic_colors = {'Crossbred': '#3498db', 'Indicine': '#2ecc71', 'Taurine': '#e74c3c'}
    breed_colors = plt.cm.Set3(np.linspace(0, 1, len(data['Breed'].unique())))
    
    for row, group_col in enumerate(group_cols):
        for col, value_col in enumerate(value_cols):
            ax = axes[row][col]
            
            # Calculate statistics
            stats_df = data.groupby(group_col)[value_col].agg(['mean', 'std', 'count']).reset_index()
            stats_df['se'] = stats_df['std'] / np.sqrt(stats_df['count'])
            
            # Set colors
            if group_col == 'Genetic Group':
                colors = [genetic_colors[x] for x in stats_df[group_col]]
            else:
                colors = breed_colors
            
            # Create bar plot
            bars = ax.bar(range(len(stats_df)), stats_df['mean'], 
                         yerr=stats_df['std'], capsize=5,
                         color=colors, alpha=0.7)
            
            # Set x-ticks
            ax.set_xticks(range(len(stats_df)))
            if group_col == 'Breed':
                ax.set_xticklabels(stats_df[group_col], rotation=45, ha='right')
            else:
                ax.set_xticklabels(stats_df[group_col])
            
            # Perform ANOVA
            f_stat, p_value, tukey = perform_statistical_analysis(data, group_col, value_col)
            
            # Create title with p-value
            gene_name = value_col.split(' Log CN')[0]  # Get just the gene name
            if p_value < 0.05:
                title = f"{gene_name} [p = {p_value:.3f}*]"
            else:
                title = f"{gene_name} [p = {p_value:.3f}]"
            
            # Customize plot
            ax.set_title(title, pad=20)
            ax.set_xlabel('')
            ax.set_ylabel('Log CN' if col == 0 else '')  # Only show y-label for first column
            
            # Set y-axis limits
            if 'HSFY' in value_col:
                ax.set_ylim(0, 6)
            elif 'ZNF280BY' in value_col:
                ax.set_ylim(0, 10)
            elif 'DDX3Y' in value_col:
                ax.set_ylim(0, 14)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    plt.close()

def save_descriptive_stats(data, group_cols, value_cols, output_file):
    with open(output_file, 'w') as f:
        for group_col in group_cols:
            f.write(f"\nDescriptive Statistics by {group_col}\n")
            f.write("=" * 50 + "\n")
            
            for value_col in value_cols:
                f.write(f"\nStatistics for {value_col}:\n")
                stats_df = data.groupby(group_col)[value_col].describe()
                f.write(stats_df.to_string())
                f.write("\n")
                
                f_stat, p_value, tukey = perform_statistical_analysis(data, group_col, value_col)
                
                f.write(f"\nOne-way ANOVA results:\n")
                f.write(f"F-statistic: {f_stat:.4f}\n")
                f.write(f"p-value: {p_value:.4f}\n")
                
                f.write("\nTukey's HSD results:\n")
                f.write(str(tukey))
                f.write("\n" + "=" * 50 + "\n")

def main():
    # Read the data
    df = pd.read_csv('CASA_data.txt', sep='\t')
    
    # Define columns for analysis
    group_cols = ['Genetic Group', 'Breed']
    value_cols = ['HSFY Log CN', 'ZNF280BY Log CN', 'DDX3Y Log CN']
    
    save_descriptive_stats(df, group_cols, value_cols, 'gene_analysis_stats.txt')
    create_plots(df, group_cols, value_cols, 'gene_analysis_plots.png')

if __name__ == "__main__":
    main()
