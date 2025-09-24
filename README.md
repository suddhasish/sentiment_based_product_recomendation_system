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