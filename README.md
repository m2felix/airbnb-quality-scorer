# 🏠 Airbnb Listing Quality Scorer

A multimodal machine learning system that predicts the quality score of an Airbnb listing based on tabular, and NLP features — with SHAP-powered explainability showing exactly what drives each prediction.

🔗 **[Live Demo](https://m2felix-airbnb-quality-scorer.streamlit.app)**

---

## What it does
A host enters their listing details — price, availability, description, host stats — and the model predicts a quality rating score and explains which features pushed it up or down.

---

## Why it's interesting
Most rating predictors are black boxes. This one uses SHAP values to surface *why* a listing scores the way it does — making it actionable, not just predictive.

---

## ML Pipeline
1. **Data** — 10,962 real San Diego listings from [Inside Airbnb](http://insideairbnb.com)
2. **Cleaning** — handled missing values, parsed price/percentage/boolean fields
3. **Features** — tabular listing features + 50 TF-IDF text features from listing descriptions
4. **Model** — Random Forest Regressor (outperformed XGBoost on this dataset)
5. **Explainability** — SHAP TreeExplainer shows per-prediction feature contributions
6. **Deployment** — Streamlit app deployed on Streamlit Cloud

---

## Key findings
- `host_is_superhost` was the single most predictive feature
- Host listing count and number of reviews followed closely
- NLP features from listing descriptions contributed meaningfully to predictions

---

## Tech stack
- Python, pandas, numpy
- scikit-learn, XGBoost
- SHAP
- Streamlit
- Inside Airbnb dataset

---

## Run locally
```bash
git clone https://github.com/m2felix/airbnb-quality-scorer
cd airbnb-quality-scorer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## What I'd add next
- Image quality features using CLIP embeddings
- More cities for generalization
- Better model with hyperparameter tuning