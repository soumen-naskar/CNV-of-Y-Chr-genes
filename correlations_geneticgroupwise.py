import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Read the data
df = pd.read_csv('CASA_data.txt', sep='\t')

# Define traits and genes
genes = ['HSFY Log CN', 'ZNF280BY Log CN', 'DDX3Y Log CN']
traits = ['Total motility [%]', 'Progressive motility [%]', 'Rapid motility [%]', 
         'Slow motility [%]', 'Local motility [%]', 'Immotile sperm [%]', 
         'Acrosome defect [%]', 'Total composite score [%]', 
         'Progressive composite score [%]', 'Sperm Viability [%]']

def create_correlation_heatmap(data, genetic_group, output_file_prefix):
    # Select only numeric columns for correlation
    variables = genes + traits
    numeric_data = data[variables].astype(float)
    
    # Calculate correlation matrix
    corr_matrix = numeric_data.corr(method='pearson')
    
    # Calculate p-values matrix
    pvalue_matrix = pd.DataFrame(np.zeros_like(corr_matrix), 
                                columns=corr_matrix.columns, 
                                index=corr_matrix.index)
    
    for i in corr_matrix.columns:
        for j in corr_matrix.index:
            if i != j:
                stat, pval = stats.pearsonr(numeric_data[i], numeric_data[j])
                pvalue_matrix.loc[i,j] = pval
    
    # Create heatmap
    plt.figure(figsize=(15, 12))
    mask = np.triu(np.ones_like(corr_matrix), k=0)  # Mask for the upper triangle
    
    # Plot heatmap
    sns.heatmap(corr_matrix, 
                annot=True,
                cmap='RdYlBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                fmt='.2f',
                square=True,
                mask=mask,
                cbar_kws={'label': 'Pearson Correlation'})
    
    # Add asterisks for significant correlations
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.index)):
            if i > j:  # Lower triangle only
                if pvalue_matrix.iloc[i,j] < 0.05:
                    plt.text(j, i, '*', 
                            ha='center', va='center', 
                            color='black')
    
    # Customize plot
    plt.title(f'Correlation Heatmap - {genetic_group}', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f'{output_file_prefix}_{genetic_group.lower()}.png', dpi=600, bbox_inches='tight')
    plt.close()
    
    # Save correlation summary to text file
    with open(f'{output_file_prefix}_{genetic_group.lower()}_summary.txt', 'w') as f:
        f.write(f"Correlation Summary for {genetic_group}\n")
        f.write("=" * 50 + "\n\n")
        
        # Write gene correlations
        f.write("Gene-Gene Correlations:\n")
        f.write("-" * 30 + "\n")
        for i, gene1 in enumerate(genes):
            for gene2 in genes[i+1:]:
                corr = corr_matrix.loc[gene1, gene2]
                pval = pvalue_matrix.loc[gene1, gene2]
                sig = '*' if pval < 0.05 else ''
                f.write(f"{gene1} vs {gene2}: r = {corr:.3f} (p = {pval:.3f}){sig}\n")
        f.write("\n")
        
        # Write gene-trait correlations
        f.write("Gene-Trait Correlations:\n")
        f.write("-" * 30 + "\n")
        for gene in genes:
            f.write(f"\n{gene} correlations:\n")
            for trait in traits:
                corr = corr_matrix.loc[gene, trait]
                pval = pvalue_matrix.loc[gene, trait]
                sig = '*' if pval < 0.05 else ''
                f.write(f"  vs {trait}: r = {corr:.3f} (p = {pval:.3f}){sig}\n")
        f.write("\n")

def main():
    # Create heatmaps for each genetic group
    for genetic_group in ['Indicine', 'Taurine', 'Crossbred']:
        group_data = df[df['Genetic Group'] == genetic_group]
        create_correlation_heatmap(group_data, genetic_group, 'correlation')

if __name__ == "__main__":
    main()
