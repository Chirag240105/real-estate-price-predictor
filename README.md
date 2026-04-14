# 🏠 Real Estate Price Prediction System

An interactive machine learning dashboard built with Streamlit that performs end-to-end real estate price prediction — from raw data to ensemble model inference — with rich Plotly visualizations and dynamic dataset support.

---

## 🚀 Live Demo

Deployed on Streamlit Cloud. Upload your own CSV or use the default `cleaned_data.csv` to get started.

---

## 📋 Features

- Full 9-step ML pipeline from data loading to model evaluation
- Ensemble modeling with 6 algorithms + a Voting Regressor
- Interactive EDA with correlation heatmaps, distributions, and box plots
- K-Fold cross-validation with score distribution plots
- Real-time price predictions with INR/USD/Lakhs currency support
- Dynamic dataset loading via file upload or local path
- Auto-detects target column, feature types, and price units

---

## 🤖 Algorithms Used

### 1. Random Forest Regressor
An ensemble of decision trees trained on random subsets of data and features. Reduces overfitting through bagging and averaging. Used as the primary model and for feature importance ranking.

### 2. Gradient Boosting Regressor
Builds trees sequentially where each tree corrects the errors of the previous one. Minimizes a loss function using gradient descent in function space. Strong performer on tabular data.

### 3. AdaBoost Regressor
Adaptive Boosting — trains weak learners (shallow trees) iteratively, giving more weight to previously misclassified samples. Combines them into a strong regressor.

### 4. Ridge Regression
Linear regression with L2 regularization. Penalizes large coefficients to reduce model variance and handle multicollinearity. Good baseline for linear relationships.

### 5. Lasso Regression
Linear regression with L1 regularization. Shrinks some coefficients to exactly zero, effectively performing feature selection alongside regression.

### 6. Decision Tree Regressor
A single tree that recursively splits data based on feature thresholds to minimize prediction error. Interpretable but prone to overfitting without pruning.

### 7. Voting Regressor (Ensemble)
Combines predictions from Random Forest, Gradient Boosting, and AdaBoost by averaging their outputs. Reduces individual model bias and variance for more robust predictions.

---

## 🔄 ML Pipeline Steps

1. Data Loading — CSV via upload, local path, or default dataset
2. Exploratory Data Analysis — statistics, correlation matrix
3. Data Cleaning & Engineering — missing value imputation, feature creation
4. Feature Selection — auto-detects target, one-hot encodes categoricals
5. Train/Test Split — 80/20 stratified split
6. Model Initialization — all 6 algorithms configured
7. Model Training — fit on scaled training data
8. K-Fold Cross-Validation — 5-fold CV with R² scoring
9. Evaluation — R², RMSE, MAE on held-out test set

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| App framework | Streamlit |
| ML | Scikit-learn |
| Data | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Stats | Statsmodels |
| Serialization | Joblib |

---

## 📂 Project Structure

```
real-estate-price-predictor/
├── app.py                   # Streamlit dashboard
├── real_estate_pipeline.py  # ML pipeline class
├── cleaned_data.csv         # Default dataset
├── scaler.pkl               # Fitted StandardScaler
├── feature_cols.pkl         # Training feature columns
├── columns.pkl              # Column reference
└── requirements.txt         # Python dependencies
```

---

## ⚙️ Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/Chirag240105/real-estate-price-predictor.git
cd real-estate-price-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

---

## 🔮 Making Predictions

1. Go to Home → click "Initialize ML Pipeline"
2. Go to Model Training → click "Train All Models"
3. Go to Predictions → enter property details → click "Predict Price"

The app supports both Bengaluru-style datasets (location/BHK/sqft) and generic datasets (area, bedrooms, bathrooms, etc.) — it auto-detects the format.

---

## 👥 Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/ayushpandey28">
        <img src="https://github.com/ayushpandey28.png" width="80px" style="border-radius:50%"/><br/>
        <b>Ayush Pandey</b>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Chirag240105">
        <img src="https://github.com/Chirag240105.png" width="80px" style="border-radius:50%"/><br/>
        <b>Chirag Pandey</b>
      </a>
    </td>
  </tr>
</table>

---

## 🎓 Project Info

CA-2 Data Science Project Exhibition — April 2026
