import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load image embeddings and ratings
embeddings = np.load('image_embeddings.npy')
ratings = pd.read_csv('image_ratings.csv')

print(f"Image embeddings: {embeddings.shape}")
print(f"Ratings: {ratings.shape}")

# Load full feature matrix and align with our 1874 valid listings
df_features = pd.read_csv('listings_features.csv')
df_clean = pd.read_csv('listings_clean.csv')
df_clean = df_clean[['picture_url', 'review_scores_rating']].dropna()
df_sample = df_clean.sample(2000, random_state=42).reset_index(drop=True)

# Get the valid indices that had successful embeddings
valid_mask = []
emb_idx = 0
for i in range(len(df_sample)):
    url = df_sample.iloc[i]['picture_url']
    try:
        import requests
        r = requests.head(url, timeout=3)
        valid_mask.append(r.status_code == 200)
    except:
        valid_mask.append(False)

# Simpler approach — just use embeddings + ratings directly
y = ratings['review_scores_rating'].values

# Baseline — tabular only (random sample of same size)
df_tab = df_features.sample(len(ratings), random_state=42)
X_tab = df_tab.drop(columns=['review_scores_rating']).values
y_tab = df_tab['review_scores_rating'].values

X_train, X_test, y_train, y_test = train_test_split(X_tab, y_tab, test_size=0.2, random_state=42)
rf_tab = RandomForestRegressor(n_estimators=300, max_depth=30, min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1)
rf_tab.fit(X_train, y_train)
preds_tab = rf_tab.predict(X_test)
print(f"\nTabular only  → RMSE: {mean_squared_error(y_test, preds_tab)**0.5:.4f} | R²: {r2_score(y_test, preds_tab):.4f}")

# Image only
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(embeddings, y, test_size=0.2, random_state=42)
rf_img = RandomForestRegressor(n_estimators=300, max_depth=30, min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1)
rf_img.fit(X_train_img, y_train_img)
preds_img = rf_img.predict(X_test_img)
print(f"Image only    → RMSE: {mean_squared_error(y_test_img, preds_img)**0.5:.4f} | R²: {r2_score(y_test_img, preds_img):.4f}")

# Combined — tabular + image
X_combined = np.hstack([X_tab, embeddings])
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_combined, y, test_size=0.2, random_state=42)
rf_combined = RandomForestRegressor(n_estimators=300, max_depth=30, min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1)
rf_combined.fit(X_train_c, y_train_c)
preds_c = rf_combined.predict(X_test_c)
print(f"Tabular+Image → RMSE: {mean_squared_error(y_test_c, preds_c)**0.5:.4f} | R²: {r2_score(y_test_c, preds_c):.4f}")