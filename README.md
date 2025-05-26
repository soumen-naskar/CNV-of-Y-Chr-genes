## Statistical Analysis of Semen Traits and Gene Copy Number Variation (CNV)
## Overview
This repository contains the complete code and pipeline used for the comprehensive statistical analysis of semen traits and their association with gene copy number variations (CNV) across different breeds and genetic backgrounds. The analyses were performed using Python 3.12 and relevant bioinformatics/statistical software and libraries.

## Analyses Performed
**1. Distribution Analysis**
The distribution of various semen traits was analyzed using the HistoBox pipeline, a robust tool for visualization and exploratory data analysis.
HistoBox repository

**2. Correlation and Variance Analysis**
Pearson correlation between CNV of genes and semen traits was calculated to assess linear relationships.

**3. Clustering Analysis**
Unsupervised machine learning clustering was conducted using t-Distributed Stochastic Neighbor Embedding (t-SNE) to group breeds based on CNV patterns. The analysis was implemented using Orange software (Demšar et al., 2013).

**4. Association Studies using General Linear Models (GLM)**
Association between semen traits and CNV was studied using GLM

**5. Regression Analysis via Random Forest**
A regression model using Random Forest (RF) was developed to predict progressive motility based on CNV features of genes. The dataset was split into 80% training and 20% test subsets. Model performance metrics include Mean Squared Error (MSE) and coefficient of determination (R²). 
The RF model can be expressed as:

**Software and Dependencies** (For all the aforesaid analyses)
Following libraries used in Python 3.12
>SciPy 
>statsmodels
>scikit-learn 
