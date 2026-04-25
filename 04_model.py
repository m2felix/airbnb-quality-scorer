import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

df = pd.read_csv('listings_features.csv')

# Separate target from features
X = df.drop(columns=['review_scores_rating'])
y = df['review_scores_rating']

print(f"Features: {X.shape[1]}")
print(f"Target range: {y.min():.2f} to {y.max():.2f}")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining on {len(X_train)} listings, testing on {len(X_test)}")

# --- Baseline: Random Forest ---
print("\nTraining Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_rmse = mean_squared_error(y_test, rf_preds) ** 0.5
rf_r2 = r2_score(y_test, rf_preds)
print(f"Random Forest → RMSE: {rf_rmse:.4f} | R²: {rf_r2:.4f}")

# --- XGBoost ---
print("\nTraining XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_rmse = mean_squared_error(y_test, xgb_preds) ** 0.5
xgb_r2 = r2_score(y_test, xgb_preds)
print(f"XGBoost      → RMSE: {xgb_rmse:.4f} | R²: {xgb_r2:.4f}")

# Save the better model
import joblib
if xgb_rmse < rf_rmse:
    joblib.dump(xgb_model, 'model.pkl')
    print("\nXGBoost saved as model.pkl")
else:
    joblib.dump(rf, 'model.pkl')
    print("\nRandom Forest saved as model.pkl")