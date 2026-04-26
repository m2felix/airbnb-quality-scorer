import pandas as pd
import numpy as np
import requests
import torch
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import os

# Load cleaned data with picture URLs
df = pd.read_csv('listings_clean.csv')
df = df[['picture_url', 'review_scores_rating']].dropna()

# Work with a sample of 2000 listings
df = df.sample(2000, random_state=42).reset_index(drop=True)

print(f"Working with {len(df)} listings")
print("Loading CLIP model...")

# Load CLIP — this downloads ~600MB the first time, cached after
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

print("CLIP model loaded. Extracting image features...")

def get_image_embedding(url):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model.vision_model(**inputs)
            embedding = outputs.pooler_output
        return embedding.squeeze().numpy()
    except:
        return None

embeddings = []
valid_indices = []

for i, row in df.iterrows():
    emb = get_image_embedding(row['picture_url'])
    if emb is not None:
        embeddings.append(emb)
        valid_indices.append(i)
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} / {len(df)} images...")

print(f"\nSuccessfully extracted {len(embeddings)} image embeddings")

# Save embeddings and corresponding ratings
embeddings_array = np.array(embeddings)
np.save('image_embeddings.npy', embeddings_array)

# Save the valid indices and ratings
df_valid = df.iloc[valid_indices][['review_scores_rating']].reset_index(drop=True)
df_valid.to_csv('image_ratings.csv', index=False)

print(f"Saved embeddings shape: {embeddings_array.shape}")
print("Done — saved to image_embeddings.npy and image_ratings.csv")