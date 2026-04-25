import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv('listings_features.csv')

X = df.drop(columns=['review_scores_rating'])
y = df['review_scores_rating']

# Load our saved model
model = joblib.load('model.pkl')

# Sample 500 rows to make SHAP faster
X_sample = X.sample(500, random_state=42)

# Build SHAP explainer
print("Building SHAP explainer...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
print("Done. Plotting feature importance...")

# Plot top 20 most important features
shap.summary_plot(
    shap_values,
    X_sample,
    max_display=20,
    show=False
)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved to shap_summary.png")

# Print top 10 features by mean SHAP value
mean_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=X_sample.columns
).sort_values(ascending=False)

print("\nTop 10 most important features:")
print(mean_shap.head(10))