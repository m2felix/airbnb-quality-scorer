import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('listings.csv')

# Check our target variable
print("Target variable stats:")
print(df['review_scores_rating'].describe())
print("\nMissing values in target:", df['review_scores_rating'].isna().sum())

# Keep only the columns we actually care about
cols_to_keep = [
    # Target
    'review_scores_rating',
    
    # Tabular features
    'price', 'accommodates', 'bedrooms', 'beds', 'bathrooms_text',
    'minimum_nights', 'availability_365', 'number_of_reviews',
    'host_is_superhost', 'host_response_rate', 'host_acceptance_rate',
    'host_identity_verified', 'instant_bookable',
    'calculated_host_listings_count', 'room_type', 'neighbourhood_cleansed',
    
    # Text feature
    'description',
    
    # Image feature
    'picture_url'
]

df = df[cols_to_keep]

print("\nNew shape:", df.shape)
print("\nMissing values per column:")
print(df.isnull().sum())