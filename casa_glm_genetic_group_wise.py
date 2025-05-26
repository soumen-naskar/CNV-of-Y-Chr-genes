import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from datetime import datetime

# Read and prepare data
data = pd.read_csv('CASA_data.txt', sep='\t')
le = LabelEncoder()
data['Genetic_Group_Encoded'] = le.fit_transform(data['Genetic Group'])

# Define traits
traits = ['Total motility [%]', 'Progressive motility [%]', 'Rapid motility [%]', 
         'Slow motility [%]', 'Local motility [%]', 'Immotile sperm [%]', 
         'Acrosome defect [%]', 'Sperm Viability [%]']

def run_glm(data, trait):
    X = data[['HSFY Log CN', 'ZNF280BY Log CN', 'DDX3Y Log CN', 'Genetic_Group_Encoded']]
    y = data[trait]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

# Part 1: Create Visualization
# Create figure with extra space for legend
fig = plt.figure(figsize=(20, 18), dpi=600)
plt.rcParams.update({'font.size': 12})

# Create color map for genetic groups
colors = {'Indicine': '#1f77b4', 'Crossbred': '#2ca02c', 'Taurine': '#ff7f0e'}

# Create subplots
for idx, trait in enumerate(traits, 1):
    ax = plt.subplot(3, 3, idx)
    
    # Get model results
    model = run_glm(data, trait)
    p_value = model.pvalues['Genetic_Group_Encoded']
    coef = model.params['Genetic_Group_Encoded']
    
    # Plot points for each genetic group
    for group in ['Indicine', 'Crossbred', 'Taurine']:
        group_data = data[data['Genetic Group'] == group]
        ax.scatter(group_data['Genetic_Group_Encoded'], group_data[trait], 
                  alpha=0.6, label=group if idx == 1 else "", color=colors[group], s=50)
    
    # Add regression line
    x_range = np.array([min(data['Genetic_Group_Encoded']), max(data['Genetic_Group_Encoded'])])
    y_pred = coef * x_range + np.mean(data[trait])
    ax.plot(x_range, y_pred, 'k--', alpha=0.8, linewidth=2)
    
    # Add significance annotation
    sig_text = f'p = {p_value:.3f}'
    if p_value < 0.001:
        sig_text += ' ***'
    elif p_value < 0.01:
        sig_text += ' **'
    elif p_value < 0.05:
        sig_text += ' *'
    
    ax.text(0.05, 0.95, sig_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    # Customize plot
    ax.set_title(trait.replace(' [%]', ''), fontsize=14, pad=10)
    ax.set_xlabel('Genetic Group', fontsize=12)
    ax.set_ylabel('Value (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Indicine', 'Crossbred', 'Taurine'], rotation=45)

# Remove empty subplot
plt.delaxes(plt.subplot(3, 3, 9))

# Add legend below the plots
handles, labels = plt.gca().get_legend_handles_labels()
fig.legend(handles, labels, 
          loc='center',
          bbox_to_anchor=(0.5, 0.02),
          ncol=3,
          fontsize=12, 
          title='Genetic Groups', 
          title_fontsize=14)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# Save the plot
plt.savefig('trait_regressions_grid.png', dpi=600, bbox_inches='tight')
plt.close()

# Part 2: Create Summary File
with open('GLM_analysis_summary.txt', 'w') as f:
    # Write header
    f.write("GLM Analysis Summary for Sperm Parameters\n")
    f.write("=" * 50 + "\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("Model: Trait = β0 + β1(HSFY) + β2(ZNF280BY) + β3(DDX3Y) + β4(Genetic Group) + ε\n\n")
    
    # Analyze each trait
    for trait in traits:
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"\nTrait: {trait}\n")
        f.write("-" * (len(trait) + 7) + "\n\n")
        
        model = run_glm(data, trait)
        
        # Model summary statistics
        f.write("Model Statistics:\n")
        f.write(f"R-squared: {model.rsquared:.3f}\n")
        f.write(f"Adjusted R-squared: {model.rsquared_adj:.3f}\n")
        f.write(f"F-statistic: {model.fvalue:.3f}\n")
        f.write(f"F-test p-value: {model.f_pvalue:.4f}\n\n")
        
        # Coefficient estimates
        f.write("Coefficient Estimates:\n")
        f.write("-" * 20 + "\n")
        
        for idx, coef_name in enumerate(['Intercept', 'HSFY Log CN', 'ZNF280BY Log CN', 
                                       'DDX3Y Log CN', 'Genetic_Group_Encoded']):
            coef = model.params[idx]
            std_err = model.bse[idx]
            p_val = model.pvalues[idx]
            
            # Add significance stars
            stars = ''
            if p_val < 0.001:
                stars = '***'
            elif p_val < 0.01:
                stars = '**'
            elif p_val < 0.05:
                stars = '*'
            
            f.write(f"{coef_name}:\n")
            f.write(f"  Coefficient: {coef:.3f} ± {std_err:.3f} {stars}\n")
            f.write(f"  p-value: {p_val:.4f}\n")
            f.write(f"  95% CI: [{model.conf_int().iloc[idx, 0]:.3f}, {model.conf_int().iloc[idx, 1]:.3f}]\n\n")
        
        # Effect sizes
        f.write("Effect Sizes:\n")
        f.write("-" * 12 + "\n")
        for idx, var in enumerate(['HSFY Log CN', 'ZNF280BY Log CN', 'DDX3Y Log CN', 'Genetic_Group_Encoded']):
            std_effect = model.params[idx + 1] * data[var].std() / data[trait].std()
            f.write(f"{var}: {std_effect:.3f} SD\n")
        f.write("\n")

    # Add note about significance levels
    f.write("\n" + "=" * 50 + "\n")
    f.write("\nSignificance levels:\n")
    f.write("* p < 0.05\n")
    f.write("** p < 0.01\n")
    f.write("*** p < 0.001\n")
    
    # Add analysis notes
    f.write("\nAnalysis Notes:\n")
    f.write("- Genetic Group was encoded as categorical variable\n")
    f.write("- Standard errors are heteroscedasticity-robust\n")
    f.write("- Effect sizes are standardized (in standard deviation units)\n")

print("Analysis complete:")
print("1. Visualization saved as 'trait_regressions_grid.png'")
print("2. Summary statistics saved as 'GLM_analysis_summary.txt'")
