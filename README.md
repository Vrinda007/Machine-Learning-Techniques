# Exploring Machine Learning Techniques in Diverse Sectors

## Project Overview
This project applies various machine learning models to analyze datasets from three distinct domains:
- **Economic Census** (Enterprise classification)
- **Housing Price Prediction** (Real estate market analysis)
- **Spotify Reviews** (Sentiment analysis and user engagement prediction)

The objective is to compare different ML models for predictive analysis and extract meaningful insights. The models implemented include Gradient Boosting, Random Forest, Linear Regression, Decision Trees, Naïve Bayes, XGBoost, and Logistic Regression.

## Datasets
1. **Economic Census Dataset:**
   - 353,015 records from six metropolitan regions (2012-2013).
   - Used for predicting enterprise types based on features like workforce distribution, ownership, and broad activity categories.
   - Models Used: Random Forest, Gradient Boosting, k-Nearest Neighbors (KNN).

2. **Housing Price Dataset:**
   - 326,876 records related to US housing prices.
   - Focuses on predicting price fluctuations based on factors like affordability, transportation, crime rate, and amenities.
   - Models Used: Linear Regression, Decision Trees, Random Forest.

3. **Spotify Reviews Dataset:**
   - 64,281 user reviews of the Spotify app.
   - Tasks: Sentiment classification (positive, neutral, negative) and predicting user engagement.
   - Models Used: Naïve Bayes (sentiment analysis), XGBoost and Logistic Regression (user engagement).

## Research Questions
- **Economic Census:** What features impact enterprise classification the most?
- **Housing Prices:** Which factors most influence property prices?
- **Spotify Analysis:** Can sentiment analysis predict user satisfaction? Can user engagement be predicted based on review frequency?

## Technologies Used
- **Python** (Data Processing & Model Training)
  - Pandas, NumPy, Scikit-learn
  - Matplotlib, Seaborn (Data Visualization)
  - XGBoost, TensorFlow (Advanced ML)
- **Jupyter Notebook** (Model Development & Experimentation)

## Data Preprocessing
- Handling missing values and outliers using Interquartile Range (IQR) and median imputation.
- Feature selection via Variance Inflation Factor (VIF) to address multicollinearity.
- Encoding categorical variables using One-Hot Encoding and TF-IDF for text processing.
- Splitting datasets into training (80%) and testing (20%).

## Model Evaluation
Models were evaluated based on accuracy, precision, recall, F1-score, RMSE, and AUC-ROC:
- **Economic Census:** Random Forest performed best with an **86.07% accuracy** and strong classification balance.
- **Housing Prices:** Random Forest achieved the **highest R² of 0.986** and lowest RMSE, outperforming other models.
- **Spotify Sentiment Analysis:** Naïve Bayes yielded an **accuracy of 63.89%**, with SMOTE applied to balance classes.
- **User Engagement Prediction:** XGBoost was the most effective model, reaching **96.18% accuracy**, outperforming Logistic Regression.

## Key Findings
- Economic census classification benefited from Random Forest’s ability to handle high-dimensional data.
- Housing price prediction relied on crime rate, transport accessibility, and amenities as major price influencers.
- Spotify sentiment analysis showed a **96% correlation** between sentiment and review scores.
- XGBoost was the **best predictor for repeated user engagement**, indicating strong consumer retention patterns.

## Future Work
- Implement deep learning models for improved prediction accuracy.
- Incorporate real-time data updates for dynamic analysis.
- Optimize hyperparameter tuning for enhanced model performance.

## Setup & Execution
### Prerequisites
- Python 3.x
- Jupyter Notebook
- Install dependencies: `pip install -r requirements.txt`

### Running the Project
1. Clone the repository: `git clone <repo-url>`
2. Open Jupyter Notebook and navigate to project folder.
3. Run preprocessing scripts:
   ```sh
   python preprocess.py
   ```
4. Train and evaluate models:
   ```sh
   python train_models.py
   ```
5. Visualize results with interactive graphs:
   ```sh
   python dashboard.py
   ```

