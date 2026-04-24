import pandas as pd
import numpy as np

df = pd.read_csv('listings.csv')

# Keep only columns we need
cols_to_keep = [
    'review_scores_rating',
    'price', 'accommodates', 'bedrooms', 'beds', 'bathrooms_text',
    'minimum_nights', 'availability_365', 'number_of_reviews',
    'host_is_superhost', 'host_response_rate', 'host_acceptance_rate',
    'host_identity_verified', 'instant_bookable',
    'calculated_host_listings_count', 'room_type', 'neighbourhood_cleansed',
    'description', 'picture_url'
]
df = df[cols_to_keep]

# Drop rows where target is missing
df = df.dropna(subset=['review_scores_rating'])
print(f"After dropping missing targets: {len(df)} listings")

# Clean price — remove $ and commas, convert to float
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

# Fill missing numerics with median
for col in ['price', 'bedrooms', 'beds']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# Clean bathrooms_text — extract the number
df['bathrooms'] = df['bathrooms_text'].str.extract(r'(\d+\.?\d*)').astype(float)
df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].median())
df = df.drop(columns=['bathrooms_text'])

# Convert yes/no and t/f columns to 1/0
df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0}).fillna(0)
df['host_identity_verified'] = df['host_identity_verified'].map({'t': 1, 'f': 0}).fillna(0)
df['instant_bookable'] = df['instant_bookable'].map({'t': 1, 'f': 0}).fillna(0)

# Clean percentage columns FIRST, then fill median
df['host_response_rate'] = df['host_response_rate'].str.replace('%', '', regex=False).astype(float)
df['host_acceptance_rate'] = df['host_acceptance_rate'].str.replace('%', '', regex=False).astype(float)
df['host_response_rate'] = df['host_response_rate'].fillna(df['host_response_rate'].median())
df['host_acceptance_rate'] = df['host_acceptance_rate'].fillna(df['host_acceptance_rate'].median())

# Fill missing description with empty string
df['description'] = df['description'].fillna('')

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['room_type', 'neighbourhood_cleansed'], drop_first=True)

print(f"Final shape: {df.shape}")
print(f"\nMissing values remaining:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# Save cleaned data
df.to_csv('listings_clean.csv', index=False)
print("\nSaved to listings_clean.csv")