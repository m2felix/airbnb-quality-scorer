import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

df = pd.read_csv('listings_features.csv')

X = df.drop(columns=['review_scores_rating'])
y = df['review_scores_rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Current baseline
print("Current baseline:")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, preds) ** 0.5:.4f} | R²: {r2_score(y_test, preds):.4f}")

# Hyperparameter search
print("\nSearching for better hyperparameters...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    scoring='r2',
    random_state=42,
    verbose=1
)

search.fit(X_train, y_train)

print(f"\nBest params: {search.best_params_}")
best_model = search.best_estimator_
best_preds = best_model.predict(X_test)
print(f"Tuned RMSE: {mean_squared_error(y_test, best_preds) ** 0.5:.4f} | R²: {r2_score(y_test, best_preds):.4f}")

# Save best model
joblib.dump(best_model, 'model.pkl')
print("\nBest model saved to model.pkl")