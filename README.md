# Sentiment-Based Product Recommendation System

**Deployed App URL:**  
https://senti-based-product-recom-95559ee11151.herokuapp.com/

## Overview
This project is an end-to-end web application that recommends products to users based on their past ratings.  
It uses a **Compact User–User Collaborative Filtering model** to generate recommendations efficiently.  
The model has been deployed as a Flask app and hosted on **Heroku**.

---

## What the App Does
- Takes a **username** as input through the web UI.  
- On submit, generates **top-5 personalized product recommendations**.  
- If no recommendations are found, shows fallback popular items.

---

## Project Structure
```
recommender_flask_project/
├─ app.py                     # Flask app
├─ model.py                   # Loads recommendation model and generates recs
├─ templates/
│  └─ index.html              # Web frontend (form + recommendations)
├─ models/
│  └─ user_user_compact.pkl   # Compact CF artifact (neighbors + ratings)
├─ requirements.txt           # Python dependencies
├─ Procfile                   # Heroku process declaration
└─ README.md                  # Project documentation
```

---

## Running Locally
1. Clone the repo and enter project folder.  
2. Create a virtual environment:  
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows
   ```
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
4. Run Flask app:  
   ```bash
   python app.py
   ```
   Visit `http://127.0.0.1:5000` in your browser.

---

## Deployment to Heroku
1. Create Heroku app:  
   ```bash
   heroku create senti-based-product-recom-95559ee11151
   ```
2. Push repo to Heroku:  
   ```bash
   git push heroku main
   ```
3. Open deployed app:  
   ```bash
   heroku open
   ```

---

## Model Details
The **Compact User–User CF model** stores only:  
- Per-user rated items and scores (`user_item_map`).  
- Top-K nearest neighbors (`user_neighbors_idx`, `user_neighbors_sim`).  
- Username normalization map (`norm_map`).  
- Item and user labels.  

✅ This reduces model size drastically (MBs instead of GBs) and is easy to deploy.

---

## Example
- Input: `Joshua`  
- Output: Top-5 recommended products with scores.

---

## Next Steps
- Add hybrid model (sentiment + CF).  
- Implement Approximate Nearest Neighbors for scalability.  
- Enhance UI with product images.  

---

---

### Project Workflow Summary

1. **Problem Statement:** Build a sentiment-driven recommendation system for an e-commerce platform using product reviews and ratings.
2. **Data Preparation:**
   - Handled missing values (dropped highly sparse columns, imputed where needed).
   - Combined `reviews_title` and `reviews_text` into a single feature (`text_for_model`).
   - Standardized target labels (`Positive` / `Negative`).
3. **Exploratory Data Analysis (EDA):**
   - Visualized missing values, review length distribution, sentiment distribution.
   - Examined product categories, brands, and metadata distributions.
4. **Feature Engineering:**
   - Applied text preprocessing (cleaning, tokenization, stopword removal).
   - Extracted features using **TF-IDF vectorization**.
5. **Model Training & Evaluation:**
   - Tested multiple classifiers (Logistic Regression, Random Forest, XGBoost, etc.).
   - Addressed class imbalance via resampling/weighting strategies.
   - Selected Logistic Regression with TF-IDF as best-performing sentiment classifier.
6. **Recommendation Systems:**
   - Implemented **User-User Collaborative Filtering** (compact neighbor-based model).
   - Implemented **Item-Item Collaborative Filtering**.
   - Evaluated models using **RMSE** (User-User ≈ 2.55 better than Item-Item ≈ 3.47).
7. **Deployment Preparation:**
   - Pickled key models (`best_lr_model.pkl`, `tfidf_vectorizer.pkl`, `user_user_compact.pkl`).
   - Built Flask web app with simple UI: takes username → outputs 5 recommended products.
   - Deployed successfully on **Heroku**.

➡️ Final system combines sentiment analysis with collaborative filtering to improve product recommendations.