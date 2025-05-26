import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_create_ratios(file_path):
    """Load data and create gene ratios"""
    # Read data
    df = pd.read_csv(file_path, delimiter='\t')
    
    # Create gene ratios
    df['HSFY_ZNF_ratio'] = df['HSFY Log CN'] / df['ZNF280BY Log CN']
    df['HSFY_DDX3Y_ratio'] = df['HSFY Log CN'] / df['DDX3Y Log CN']
    df['ZNF_DDX3Y_ratio'] = df['ZNF280BY Log CN'] / df['DDX3Y Log CN']
    
    return df

def analyze_relationships(df):
    """Analyze relationships between individual genes, ratios, and traits"""
    # Define genes, ratios, and traits
    genes = ['HSFY Log CN', 'ZNF280BY Log CN', 'DDX3Y Log CN']
    ratios = ['HSFY_ZNF_ratio', 'HSFY_DDX3Y_ratio', 'ZNF_DDX3Y_ratio']
    traits = ['Total motility [%]', 'Progressive motility [%]', 'Rapid motility [%]', 
             'Slow motility [%]', 'Local motility [%]', 'Immotile sperm [%]', 
             'Acrosome defect [%]', 'Sperm Viability [%]']
    
    results = []
    
    # Analyze each trait
    for trait in traits:
        # Individual gene correlations
        gene_cors = {}
        for gene in genes:
            correlation = stats.pearsonr(df[gene], df[trait])
            gene_cors[gene] = {'r': correlation[0], 'p': correlation[1]}
        
        # Ratio correlations
        ratio_cors = {}
        for ratio in ratios:
            correlation = stats.pearsonr(df[ratio], df[trait])
            ratio_cors[ratio] = {'r': correlation[0], 'p': correlation[1]}
        
        # Store results
        results.append({
            'Trait': trait,
            'Best_Gene': max(gene_cors.items(), key=lambda x: abs(x[1]['r'])),
            'Best_Ratio': max(ratio_cors.items(), key=lambda x: abs(x[1]['r'])),
            'Gene_Correlations': gene_cors,
            'Ratio_Correlations': ratio_cors
        })
    
    return results

def plot_ratio_comparisons(df, output_prefix):
    """Create plots comparing gene ratios with traits"""
    ratios = ['HSFY_ZNF_ratio', 'HSFY_DDX3Y_ratio', 'ZNF_DDX3Y_ratio']
    traits = ['Total motility [%]', 'Progressive motility [%]', 'Rapid motility [%]']
    
    # Create plots for each trait
    for trait in traits:
        plt.figure(figsize=(15, 5))
        
        for idx, ratio in enumerate(ratios, 1):
            plt.subplot(1, 3, idx)
            
            # Scatter plot
            sns.scatterplot(data=df, x=ratio, y=trait)
            
            # Add regression line
            sns.regplot(data=df, x=ratio, y=trait, scatter=False, color='red')
            
            # Calculate correlation
            corr = stats.pearsonr(df[ratio], df[trait])
            plt.title(f'{ratio} vs {trait}\nr={corr[0]:.3f}, p={corr[1]:.3f}')
            
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_{trait.split()[0]}.png', dpi=300, bbox_inches='tight')
        plt.close()

def compare_models(df, trait):
    """Compare models with individual genes vs ratios"""
    # Prepare predictors
    genes = ['HSFY Log CN', 'ZNF280BY Log CN', 'DDX3Y Log CN']
    ratios = ['HSFY_ZNF_ratio', 'HSFY_DDX3Y_ratio', 'ZNF_DDX3Y_ratio']
    
    # Model with individual genes
    X_genes = sm.add_constant(df[genes])
    y = df[trait]
    gene_model = sm.OLS(y, X_genes).fit()
    
    # Model with ratios
    X_ratios = sm.add_constant(df[ratios])
    ratio_model = sm.OLS(y, X_ratios).fit()
    
    # Combined model
    X_combined = sm.add_constant(df[genes + ratios])
    combined_model = sm.OLS(y, X_combined).fit()
    
    return {
        'trait': trait,
        'gene_r2': gene_model.rsquared,
        'ratio_r2': ratio_model.rsquared,
        'combined_r2': combined_model.rsquared
    }

def main():
    # Load data and create ratios
    df = load_and_create_ratios('CASA_data.txt')
    
    # Analyze relationships
    results = analyze_relationships(df)
    
    # Save correlation results
    with open('gene_ratio_analysis.txt', 'w') as f:
        f.write("Gene Ratio Analysis Results\n")
        f.write("==========================\n\n")
        
        for result in results:
            f.write(f"\nTrait: {result['Trait']}\n")
            f.write("-" * 50 + "\n")
            
            # Best gene correlation
            best_gene = result['Best_Gene']
            f.write(f"Best Gene: {best_gene[0]}, r={best_gene[1]['r']:.3f}, p={best_gene[1]['p']:.3f}\n")
            
            # Best ratio correlation
            best_ratio = result['Best_Ratio']
            f.write(f"Best Ratio: {best_ratio[0]}, r={best_ratio[1]['r']:.3f}, p={best_ratio[1]['p']:.3f}\n")
            
            # All gene correlations
            f.write("\nGene Correlations:\n")
            for gene, cors in result['Gene_Correlations'].items():
                f.write(f"{gene}: r={cors['r']:.3f}, p={cors['p']:.3f}\n")
            
            # All ratio correlations
            f.write("\nRatio Correlations:\n")
            for ratio, cors in result['Ratio_Correlations'].items():
                f.write(f"{ratio}: r={cors['r']:.3f}, p={cors['p']:.3f}\n")
    
    # Create visualization plots
    plot_ratio_comparisons(df, 'ratio_comparison')
    
    # Compare models for key traits
    model_comparisons = []
    key_traits = ['Total motility [%]', 'Progressive motility [%]', 'Rapid motility [%]']
    
    for trait in key_traits:
        comparison = compare_models(df, trait)
        model_comparisons.append(comparison)
    
    # Save model comparisons
    comparison_df = pd.DataFrame(model_comparisons)
    with open('gene_ratio_analysis.txt', 'a') as f:
        f.write("\n\nModel Comparisons\n")
        f.write("================\n")
        f.write(comparison_df.to_string())

if __name__ == "__main__":
    main()
