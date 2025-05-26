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

# Perform ANOVA for each parameter by breed
def perform_anova_by_breed(data, parameter):
    breeds = data['Breed'].unique()
    
    # Prepare data for ANOVA
    data_list = [group_data[parameter].values for name, group_data in data.groupby('Breed')]
    
    # Perform one-way ANOVA
    f_statistic, p_value = stats.f_oneway(*data_list)
    
    return {
        'Parameter': parameter,
        'F-statistic': f_statistic,
        'p-value': p_value
    }

# Store ANOVA results
anova_results = []
for param in numeric_columns:
    result = perform_anova_by_breed(data, param)
    anova_results.append(result)

# Convert results to DataFrame
results_df = pd.DataFrame(anova_results)
print("\nANOVA Results by Breed:")
print(results_df)

# Create box plots
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(20, 15))  # Increased figure size for better breed label visibility
fig.suptitle('Key Parameters by Breed', fontsize=16)

params_to_plot = ['Total motility [%]', 'Progressive motility [%]', 
                 'Sperm Viability [%]', 'Acrosome defect [%]']

for ax, param in zip(axes.flat, params_to_plot):
    # Get data for each breed
    breeds = data['Breed'].unique()
    breed_data = [data[data['Breed'] == breed][param] for breed in breeds]
    
    # Create box plot
    bp = ax.boxplot(breed_data, labels=breeds)
    ax.set_xticklabels(breeds, rotation=90)  # Increased rotation for better readability
    ax.set_title(param)
    ax.set_ylabel(param)
    
    # Add individual points for better visualization
    for i, breed in enumerate(breeds, 1):
        y = data[data['Breed'] == breed][param]
        x = np.random.normal(i, 0.04, size=len(y))
        ax.plot(x, y, 'r.', alpha=0.2)

plt.tight_layout()
plt.show()

# Create violin plots
fig, axes = plt.subplots(1, 2, figsize=(20, 8))  # Increased figure size
violin_params = ['Total motility [%]', 'Sperm Viability [%]']

for ax, param in zip(axes, violin_params):
    # Get data for each breed
    breeds = data['Breed'].unique()
    breed_data = [data[data['Breed'] == breed][param] for breed in breeds]
    
    # Create violin plot
    parts = ax.violinplot(breed_data, showmeans=True, showmedians=True)
    
    # Customize violin plot
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    # Set labels and title
    ax.set_xticks(range(1, len(breeds) + 1))
    ax.set_xticklabels(breeds, rotation=90)
    ax.set_title(f'Distribution of {param} by Breed')
    ax.set_ylabel(param)

plt.tight_layout()
plt.show()

# Calculate and print summary statistics by breed
print("\nSummary Statistics by Breed:")
summary_stats = data.groupby('Breed')[numeric_columns].agg(['mean', 'std'])
print(summary_stats)

# Perform Tukey's HSD test for post-hoc analysis for all traits
for param in numeric_columns:
    print(f"\nTukey's HSD Test Results for {param} by Breed:")
    tukey = pairwise_tukeyhsd(data[param], data['Breed'])
    print(tukey)

# Additional analysis: Number of samples per breed
print("\nNumber of samples per breed:")
breed_counts = data['Breed'].value_counts()
print(breed_counts)

# Create a bar plot of sample sizes
plt.figure(figsize=(12, 6))
breed_counts.plot(kind='bar')
plt.title('Number of Samples per Breed')
plt.xlabel('Breed')
plt.ylabel('Number of Samples')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
