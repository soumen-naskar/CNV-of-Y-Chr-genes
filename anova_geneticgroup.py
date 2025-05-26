import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Read the data
data = pd.read_csv('CASA_data.txt', sep='\t')

# List of numeric columns to analyze (excluding the gene copy number columns)
numeric_columns = ['Total motility [%]', 'Progressive motility [%]', 'Rapid motility [%]', 
                  'Slow motility [%]', 'Local motility [%]', 'Immotile sperm [%]', 
                  'Acrosome defect [%]', 'Sperm Viability [%]']

# Perform two-way ANOVA for each parameter
def perform_two_way_anova(data, parameter):
    groups = data['Genetic Group'].unique()
    results = []
    
    # Prepare data for ANOVA
    data_list = [group_data[parameter].values for name, group_data in data.groupby('Genetic Group')]
    
    # Perform one-way ANOVA (since we're only considering Genetic Group)
    f_statistic, p_value = stats.f_oneway(*data_list)
    
    return {
        'Parameter': parameter,
        'F-statistic': f_statistic,
        'p-value': p_value
    }

# Store ANOVA results
anova_results = []
for param in numeric_columns:
    result = perform_two_way_anova(data, param)
    anova_results.append(result)

# Convert results to DataFrame
results_df = pd.DataFrame(anova_results)
print("\nTwo-way ANOVA Results:")
print(results_df)

# Create box plots
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Key Parameters by Genetic Group', fontsize=16)

params_to_plot = ['Total motility [%]', 'Progressive motility [%]', 
                 'Sperm Viability [%]', 'Acrosome defect [%]']

for ax, param in zip(axes.flat, params_to_plot):
    # Get data for each group
    groups = data['Genetic Group'].unique()
    group_data = [data[data['Genetic Group'] == group][param] for group in groups]
    
    # Create box plot
    bp = ax.boxplot(group_data, labels=groups)
    ax.set_xticklabels(groups, rotation=45)
    ax.set_title(param)
    ax.set_ylabel(param)
    
    # Add individual points for better visualization
    for i, group in enumerate(groups, 1):
        y = data[data['Genetic Group'] == group][param]
        x = np.random.normal(i, 0.04, size=len(y))
        ax.plot(x, y, 'r.', alpha=0.2)

plt.tight_layout()
plt.show()

# Create violin plots using matplotlib
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
violin_params = ['Total motility [%]', 'Sperm Viability [%]']

for ax, param in zip(axes, violin_params):
    # Get data for each group
    groups = data['Genetic Group'].unique()
    group_data = [data[data['Genetic Group'] == group][param] for group in groups]
    
    # Create violin plot
    parts = ax.violinplot(group_data, showmeans=True, showmedians=True)
    
    # Customize violin plot
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Set labels and title
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups, rotation=45)
    ax.set_title(f'Distribution of {param} by Genetic Group')
    ax.set_ylabel(param)

plt.tight_layout()
plt.show()

# Calculate and print summary statistics
print("\nSummary Statistics by Genetic Group:")
summary_stats = data.groupby('Genetic Group')[numeric_columns].agg(['mean', 'std'])
print(summary_stats)

# Perform Tukey's HSD test for post-hoc analysis for all traits
for param in numeric_columns:
    print(f"\nTukey's HSD Test Results for {param} by Genetic Group:")
    tukey = pairwise_tukeyhsd(data[param], data['Genetic Group'])
    print(tukey)
    
    # Add mean values for each genetic group for this parameter
    print(f"\nMean {param} by Genetic Group:")
    means = data.groupby('Genetic Group')[param].mean()
    print(means)
    print("\n" + "="*50 + "\n")
