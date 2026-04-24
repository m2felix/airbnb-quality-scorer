import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('listings_clean.csv')

print(f"Loaded {len(df)} listings")

# --- Text features from description ---
print("\nBuilding text features...")

df['description'] = df['description'].fillna('')
tfidf = TfidfVectorizer(max_features=50, stop_words='english')
text_features = tfidf.fit_transform(df['description']).toarray()
text_df = pd.DataFrame(text_features, columns=[f'tfidf_{i}' for i in range(50)])

print(f"Text features shape: {text_df.shape}")

# --- Combine with tabular features ---
# Drop columns we don't feed into the model
df = df.drop(columns=['description', 'picture_url'])

# Reset index before concat
df = df.reset_index(drop=True)
text_df = text_df.reset_index(drop=True)

# Combine
df_final = pd.concat([df, text_df], axis=1)

print(f"Final feature matrix shape: {df_final.shape}")
print(f"Missing values: {df_final.isnull().sum().sum()}")

# Save
df_final.to_csv('listings_features.csv', index=False)
print("\nSaved to listings_features.csv")