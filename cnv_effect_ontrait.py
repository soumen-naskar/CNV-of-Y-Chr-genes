import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_prepare_data(file_path):
    """Load and prepare the data for analysis"""
    df = pd.read_csv(file_path, delimiter='\t')
    
    cnv_genes = ['HSFY Log CN', 'ZNF280BY Log CN', 'DDX3Y Log CN']
    traits = ['Total motility [%]', 'Progressive motility [%]', 'Rapid motility [%]', 
             'Slow motility [%]', 'Local motility [%]', 'Immotile sperm [%]', 
             'Acrosome defect [%]', 'Total composite score [%]', 
             'Progressive composite score [%]', 'Sperm Viability [%]']
    
    return df, cnv_genes, traits

def analyze_relationship(X, y):
    """Analyze linear and quadratic relationships"""
    # Linear model
    X_const = sm.add_constant(X)
    linear_model = sm.OLS(y, X_const).fit()
    
    # Quadratic model
    X_quad = np.column_stack((X, X**2))
    X_quad = sm.add_constant(X_quad)
    quad_model = sm.OLS(y, X_quad).fit()
    
    # Calculate correlations
    pearson_r, pearson_p = stats.pearsonr(X.flatten(), y)
    spearman_r, spearman_p = stats.spearmanr(X.flatten(), y)
    
    return {
        'linear_r2': linear_model.rsquared,
        'linear_adj_r2': linear_model.rsquared_adj,
        'linear_p_value': linear_model.f_pvalue,
        'linear_coef': linear_model.params[1],
        'quad_r2': quad_model.rsquared,
        'quad_adj_r2': quad_model.rsquared_adj,
        'quad_p_value': quad_model.f_pvalue,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p
    }

def remove_outliers(df, columns, z_threshold=3):
    """Remove outliers using z-score method"""
    df_clean = df.copy()
    for col in columns:
        z_scores = np.abs(stats.zscore(df_clean[col]))
        df_clean = df_clean[z_scores < z_threshold]
    return df_clean

def standardize_variables(df, columns):
    """Standardize variables"""
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    return df_scaled, scaler

def plot_gene_relationships(df, gene, traits, filename_base):
    """Create a grid of plots and save summary statistics"""
    n_traits = len(traits)
    rows = (n_traits + 1) // 2
    
    # Create plot
    fig = plt.figure(figsize=(20, 5*rows))
    plt.suptitle(f'Relationships between {gene} and Sperm Quality Traits', fontsize=16, y=0.95)
    
    # Open text file for summary statistics
    with open(f'{filename_base}_summary.txt', 'w') as f:
        f.write(f"Summary Statistics for {gene}\n")
        f.write("=" * 50 + "\n\n")
        
        for idx, trait in enumerate(traits, 1):
            # Plot
            plt.subplot(rows, 2, idx)
            
            X = df[gene].values.reshape(-1, 1)
            y = df[trait].values
            
            # Get analysis results
            results = analyze_relationship(X, y)
            
            # Write summary statistics
            f.write(f"\nTrait: {trait}\n")
            f.write("-" * 30 + "\n")
            f.write("Linear Model:\n")
            f.write(f"R-squared: {results['linear_r2']:.4f}\n")
            f.write(f"Adjusted R-squared: {results['linear_adj_r2']:.4f}\n")
            f.write(f"P-value: {results['linear_p_value']:.4f}\n")
            f.write(f"Coefficient: {results['linear_coef']:.4f}\n")
            
            f.write("\nQuadratic Model:\n")
            f.write(f"R-squared: {results['quad_r2']:.4f}\n")
            f.write(f"Adjusted R-squared: {results['quad_adj_r2']:.4f}\n")
            f.write(f"P-value: {results['quad_p_value']:.4f}\n")
            
            f.write("\nCorrelations:\n")
            f.write(f"Pearson r: {results['pearson_r']:.4f} (p={results['pearson_p']:.4f})\n")
            f.write(f"Spearman r: {results['spearman_r']:.4f} (p={results['spearman_p']:.4f})\n")
            f.write("\n" + "=" * 50 + "\n")
            
            # Plot
            sns.scatterplot(data=df, x=gene, y=trait)
            sns.regplot(data=df, x=gene, y=trait, scatter=False, color='red', label='Linear')
            
            # Quadratic fit
            x = df[gene]
            y = df[trait]
            z = np.polyfit(x, y, 2)
            p = np.poly1d(z)
            x_new = np.linspace(x.min(), x.max(), 100)
            plt.plot(x_new, p(x_new), 'g--', label='Quadratic')
            
            plt.title(f'{gene} vs {trait}')
            plt.legend()
        
        # Adjust layout and save plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{filename_base}_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Load data
    df, cnv_genes, traits = load_and_prepare_data('CASA_data.txt')
    
    # Remove outliers
    df_clean = remove_outliers(df, cnv_genes + traits)
    
    # Standardize variables
    df_scaled, _ = standardize_variables(df_clean, cnv_genes + traits)
    
    # Create plots and save summaries for each gene
    for gene in cnv_genes:
        gene_name = gene.split()[0]
        filename_base = f'{gene_name}'
        plot_gene_relationships(df_scaled, gene, traits, filename_base)
        print(f"Generated analysis for {gene}")
        print(f"- Plot saved as {filename_base}_plots.png")
        print(f"- Summary saved as {filename_base}_summary.txt")

if __name__ == "__main__":
    main()
