import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# --- Build everything from listings.csv if needed ---
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv('listings.csv')

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
    df = df.dropna(subset=['review_scores_rating'])

    # Clean price
    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

    # Fill numerics
    for col in ['price', 'bedrooms', 'beds']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # Bathrooms
    df['bathrooms'] = df['bathrooms_text'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].median())
    df = df.drop(columns=['bathrooms_text'])

    # Boolean columns
    df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0}).fillna(0)
    df['host_identity_verified'] = df['host_identity_verified'].map({'t': 1, 'f': 0}).fillna(0)
    df['instant_bookable'] = df['instant_bookable'].map({'t': 1, 'f': 0}).fillna(0)

    # Percentage columns
    df['host_response_rate'] = df['host_response_rate'].str.replace('%', '', regex=False).astype(float)
    df['host_acceptance_rate'] = df['host_acceptance_rate'].str.replace('%', '', regex=False).astype(float)
    df['host_response_rate'] = df['host_response_rate'].fillna(df['host_response_rate'].median())
    df['host_acceptance_rate'] = df['host_acceptance_rate'].fillna(df['host_acceptance_rate'].median())

    # Description
    df['description'] = df['description'].fillna('')

    # One-hot encode
    df = pd.get_dummies(df, columns=['room_type', 'neighbourhood_cleansed'], drop_first=True)

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=50, stop_words='english')
    text_features = tfidf.fit_transform(df['description']).toarray()
    text_df = pd.DataFrame(text_features, columns=[f'tfidf_{i}' for i in range(50)])

    df = df.drop(columns=['description', 'picture_url'])
    df = df.reset_index(drop=True)
    text_df = text_df.reset_index(drop=True)
    df_final = pd.concat([df, text_df], axis=1)

    X = df_final.drop(columns=['review_scores_rating'])
    y = df_final['review_scores_rating']

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    return model, X, y, tfidf

# --- App UI ---
st.set_page_config(page_title="Airbnb Listing Quality Scorer", page_icon="🏠")
st.title("🏠 Airbnb Listing Quality Scorer")
st.markdown("Predict how well your listing will be rated based on its features.")

with st.spinner("Loading model... this takes about 30 seconds on first load."):
    model, X, y, tfidf = load_model_and_data()

st.header("Tell us about your listing")

col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Price per night ($)", min_value=10, max_value=2000, value=150)
    accommodates = st.slider("Accommodates (guests)", 1, 16, 2)
    bedrooms = st.slider("Bedrooms", 0, 10, 1)
    beds = st.slider("Beds", 1, 10, 1)
    bathrooms = st.slider("Bathrooms", 1, 10, 1)

with col2:
    minimum_nights = st.slider("Minimum nights", 1, 30, 2)
    availability_365 = st.slider("Days available per year", 0, 365, 180)
    number_of_reviews = st.number_input("Number of reviews so far", 0, 1000, 10)
    host_listings_count = st.number_input("How many listings you host", 1, 100, 1)

st.subheader("Host details")
col3, col4 = st.columns(2)
with col3:
    is_superhost = st.selectbox("Are you a superhost?", ["No", "Yes"])
    identity_verified = st.selectbox("Identity verified?", ["No", "Yes"])
with col4:
    instant_bookable = st.selectbox("Instant bookable?", ["No", "Yes"])
    host_response_rate = st.slider("Host response rate (%)", 0, 100, 90)
    host_acceptance_rate = st.slider("Host acceptance rate (%)", 0, 100, 85)

st.subheader("Listing description")
description = st.text_area("Paste your listing description here", height=150)

if st.button("Score my listing ↗"):
    text_vec = tfidf.transform([description]).toarray()
    text_df_input = pd.DataFrame(text_vec, columns=[f'tfidf_{i}' for i in range(50)])

    input_dict = {col: 0 for col in X.columns}
    input_dict['price'] = price
    input_dict['accommodates'] = accommodates
    input_dict['bedrooms'] = bedrooms
    input_dict['beds'] = beds
    input_dict['bathrooms'] = bathrooms
    input_dict['minimum_nights'] = minimum_nights
    input_dict['availability_365'] = availability_365
    input_dict['number_of_reviews'] = number_of_reviews
    input_dict['calculated_host_listings_count'] = host_listings_count
    input_dict['host_is_superhost'] = 1 if is_superhost == "Yes" else 0
    input_dict['host_identity_verified'] = 1 if identity_verified == "Yes" else 0
    input_dict['instant_bookable'] = 1 if instant_bookable == "Yes" else 0
    input_dict['host_response_rate'] = host_response_rate
    input_dict['host_acceptance_rate'] = host_acceptance_rate

    for i in range(50):
        input_dict[f'tfidf_{i}'] = text_df_input[f'tfidf_{i}'].values[0]

    input_df = pd.DataFrame([input_dict])

    score = model.predict(input_df)[0]
    score = np.clip(score, 1, 5)

    st.markdown("---")
    st.header("Your predicted quality score")

    col_score, col_context = st.columns([1, 2])
    with col_score:
        st.metric("Predicted Rating", f"{score:.2f} / 5.00")
    with col_context:
        if score >= 4.8:
            st.success("Top tier listing — likely to rank highly in search results.")
        elif score >= 4.5:
            st.info("Solid listing — small improvements could push it higher.")
        else:
            st.warning("Below average — focus on superhost status and description quality.")

    st.subheader("What's driving your score?")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)
    shap_series = pd.Series(shap_vals[0], index=X.columns)
    top_shap = shap_series.abs().sort_values(ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#2ecc71' if shap_series[f] > 0 else '#e74c3c' for f in top_shap.index]
    ax.barh(top_shap.index[::-1], shap_series[top_shap.index[::-1]], color=colors[::-1])
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Impact on score")
    ax.set_title("Feature contributions to your score")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()