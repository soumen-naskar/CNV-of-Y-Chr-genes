import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

def calculate_effect_size(coefficient, y):
    """Calculate standardized effect size in SD units"""
    sd_y = np.std(y)
    return coefficient / sd_y

def run_breed_analysis(data, trait):
    """Run one-way ANOVA for breed effect"""
    breeds = sorted(data['Breed'].unique())
    breed_groups = [group[trait].dropna().values for name, group in data.groupby('Breed')]
    if all(len(group) > 0 for group in breed_groups):
        f_stat, p_value = stats.f_oneway(*breed_groups)
        return f_stat, p_value
    return np.nan, np.nan

def run_glm_analysis(data, trait, breeds):
    """Run GLM analysis for a single trait"""
    # Ensure data is numeric
    y = pd.to_numeric(data[trait], errors='coerce')
    
    # Drop any rows with NaN values
    valid_indices = ~y.isna()
    y = y[valid_indices]
    
    # Create dummy variables for breeds
    breed_data = data.loc[valid_indices, 'Breed']
    breed_dummies = pd.get_dummies(breed_data, prefix='Breed', drop_first=True)
    
    # Ensure all data is float64
    breed_dummies = breed_dummies.astype(np.float64)
    y = y.astype(np.float64)
    
    # Add constant for intercept
    X = sm.add_constant(breed_dummies)
    
    # Fit model with robust standard errors
    try:
        model = sm.OLS(y, X)
        results = model.fit(cov_type='HC1')
        
        # Calculate effect sizes
        effect_sizes = {}
        for var in breed_dummies.columns:
            idx = results.params.index.get_loc(var)  # Using index location instead of position
            effect_sizes[var] = calculate_effect_size(results.params.iloc[idx], y)
        
        return results, effect_sizes
    except Exception as e:
        print(f"Error in GLM analysis for {trait}: {str(e)}")
        raise

def format_coefficient(coef, se, pval, ci_low, ci_high):
    """Format coefficient information"""
    stars = ''
    if pval < 0.001:
        stars = ' ***'
    elif pval < 0.01:
        stars = ' **'
    elif pval < 0.05:
        stars = ' *'
    
    return (f"  Coefficient: {coef:.3f} ± {se:.3f}{stars}\n"
            f"  p-value: {pval:.4f}\n"
            f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

def create_visualization(data, traits):
    """Create visualization of breed effects"""
    plt.figure(figsize=(20, 15), dpi=600)
    plt.rcParams.update({'font.size': 12})
    
    for idx, trait in enumerate(traits, 1):
        ax = plt.subplot(3, 3, idx)
        
        # Get breed-specific data
        breeds = sorted(data['Breed'].unique())
        breed_positions = range(len(breeds))
        
        # Plot points for each breed
        means = []
        for i, breed in enumerate(breeds):
            breed_data = data[data['Breed'] == breed]
            
            # Plot individual points
            y = breed_data[trait].dropna()
            if len(y) > 0:
                x = np.repeat(i, len(y))
                ax.scatter(x + np.random.normal(0, 0.05, len(y)), y, 
                          alpha=0.6, color='#1f77b4', s=40)
                
                # Calculate mean and std error for error bars
                mean = y.mean()
                means.append(mean)
                stderr = y.std() / np.sqrt(len(y))
                ax.errorbar(i, mean, yerr=stderr, fmt='o', color='red', 
                           markersize=8, capsize=5, capthick=2, elinewidth=2)
            else:
                means.append(np.nan)
        
        # Run ANOVA and get p-value
        f_stat, p_value = run_breed_analysis(data, trait)
        
        # Add significance annotation
        if not np.isnan(p_value):
            sig_text = f'p = {p_value:.3f}'
            if p_value < 0.001:
                sig_text += ' ***'
            elif p_value < 0.01:
                sig_text += ' **'
            elif p_value < 0.05:
                sig_text += ' *'
            
            # Add trend line if significant
            if p_value < 0.05:
                valid_means = [m for m in means if not np.isnan(m)]
                if len(valid_means) > 1:
                    valid_positions = [i for i, m in enumerate(means) if not np.isnan(m)]
                    z = np.polyfit(valid_positions, valid_means, 1)
                    p = np.poly1d(z)
                    ax.plot(valid_positions, p(valid_positions), 
                            "k--", alpha=0.8, linewidth=2)
            
            ax.text(0.05, 0.95, sig_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        # Customize plot
        ax.set_title(trait.replace(' [%]', ''), fontsize=14, pad=10)
        ax.set_xlabel('Breed', fontsize=12)
        ax.set_ylabel('Value (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set x-ticks to breed names
        ax.set_xticks(breed_positions)
        ax.set_xticklabels(breeds, rotation=45, ha='right')
    
    # Remove empty subplot
    plt.delaxes(plt.subplot(3, 3, 9))
    plt.tight_layout()
    plt.savefig('breed_effects.png', dpi=600, bbox_inches='tight')
    plt.close()

def write_glm_summary(data, traits, output_file='GLM_breedwise_summary.txt'):
    """Write GLM analysis summary"""
    breeds = sorted(data['Breed'].unique())
    reference_breed = breeds[0]
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("GLM Analysis Summary for Sperm Parameters\n")
        f.write("=" * 50 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write model specification
        f.write(f"Model: Trait = β0 + Σβi(Breedi) + ε\n")
        f.write(f"Reference Breed: {reference_breed}\n\n")
        f.write("=" * 50 + "\n")
        
        # Analyze each trait
        for trait in traits:
            print(f"Analyzing trait: {trait}")
            f.write(f"\nTrait: {trait}\n")
            f.write("-" * (len(trait) + 7) + "\n\n")
            
            try:
                results, effect_sizes = run_glm_analysis(data, trait, breeds)
                
                # Write model statistics
                f.write("Model Statistics:\n")
                f.write(f"R-squared: {results.rsquared:.3f}\n")
                f.write(f"Adjusted R-squared: {results.rsquared_adj:.3f}\n")
                f.write(f"F-statistic: {results.fvalue:.3f}\n")
                f.write(f"F-test p-value: {results.f_pvalue:.4f}\n\n")
                
                # Write coefficient estimates
                f.write("Coefficient Estimates:\n")
                f.write("--------------------\n")
                
                # Intercept (reference breed)
                ci = results.conf_int().loc['const']
                f.write("Intercept (Reference Breed):\n")
                f.write(format_coefficient(
                    results.params['const'],
                    results.bse['const'],
                    results.pvalues['const'],
                    ci[0], ci[1]
                ))
                f.write("\n\n")
                
                # Breed effects (relative to reference)
                for breed in results.params.index[1:]:
                    ci = results.conf_int().loc[breed]
                    f.write(f"{breed}:\n")
                    f.write(format_coefficient(
                        results.params[breed],
                        results.bse[breed],
                        results.pvalues[breed],
                        ci[0], ci[1]
                    ))
                    f.write("\n\n")
                
                # Write effect sizes
                f.write("Effect Sizes (in SD units):\n")
                f.write("------------\n")
                for breed, effect_size in effect_sizes.items():
                    f.write(f"{breed}: {effect_size:.3f} SD\n")
                
            except Exception as e:
                f.write(f"\nError analyzing {trait}: {str(e)}\n")
                print(f"Error analyzing {trait}: {str(e)}")
            
            f.write("\n" + "=" * 50 + "\n")
        
        # Write significance level legend
        f.write("\nSignificance levels:\n")
        f.write("* p < 0.05\n")
        f.write("** p < 0.01\n")
        f.write("*** p < 0.001\n\n")
        
        # Write analysis notes
        f.write("Analysis Notes:\n")
        f.write("- Reference breed is used as baseline for comparisons\n")
        f.write("- Standard errors are heteroscedasticity-robust (HC1)\n")
        f.write("- Effect sizes are standardized (in standard deviation units)\n")

def main():
    # Read and prepare data
    print("Reading data file...")
    data = pd.read_csv('CASA_data.txt', sep='\t')
    
    # Define traits
    traits = ['Total motility [%]', 'Progressive motility [%]', 'Rapid motility [%]', 
              'Slow motility [%]', 'Local motility [%]', 'Immotile sperm [%]', 
              'Acrosome defect [%]', 'Sperm Viability [%]']
    
    # Convert data to numeric type
    print("Converting data types...")
    for trait in traits:
        data[trait] = pd.to_numeric(data[trait], errors='coerce')
        print(f"Converted {trait}: {data[trait].dtype}")
    
    # Create visualization
    print("Creating visualization...")
    create_visualization(data, traits)
    
    # Generate GLM summary
    print("Running GLM analysis...")
    write_glm_summary(data, traits)
    
    print("Analysis complete:")
    print("1. Breed effects visualization saved as 'breed_effects.png'")
    print("2. GLM analysis summary saved as 'GLM_breedwise_summary.txt'")

if __name__ == "__main__":
    main()
