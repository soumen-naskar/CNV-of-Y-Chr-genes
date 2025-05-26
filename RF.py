import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
import numpy as np

# Load and preprocess CASA data
data = pd.read_csv("CASA_data.txt", sep='\t')

# Select features and target
selected_features = ["HSFY Log CN", "ZNF280BY Log CN", "DDX3Y Log CN",
                     "Total motility [%]", "Progressive motility [%]",
                     "Sperm Viability [%]"]
X = data[selected_features]
y = data["Progressive motility [%]"]  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save metrics and feature importance to a text file
with open("rf_summary.txt", "w") as f:
    f.write(f"Random Forest Regression Summary\n")
    f.write(f"Mean Squared Error: {mse:.4f}\n")
    f.write(f"R² Score: {r2:.4f}\n\n")
    f.write("Feature Importances:\n")
    for feature, importance in zip(selected_features, rf_model.feature_importances_):
        f.write(f"{feature}: {importance:.4f}\n")

print("Summary saved to rf_summary.txt")

# Visualize Feature Importance
importances = rf_model.feature_importances_
plt.figure(figsize=(8, 6))
plt.barh(selected_features, importances)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance")
plt.savefig("feature_importance.png")
plt.show()

# Partial Dependence Plots
fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(rf_model, X, ["HSFY Log CN", "ZNF280BY Log CN"], ax=ax)
plt.savefig("partial_dependence.png")
plt.show()

# Observed vs. Predicted Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, label="Predictions")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Ideal Fit")
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.title("Observed vs. Predicted")
plt.legend()
plt.savefig("observed_vs_predicted.png")
plt.show()

