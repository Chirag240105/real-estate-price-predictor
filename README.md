# Real Estate Price Prediction System

An interactive Streamlit dashboard that performs exploratory data analysis, trains multiple ML models, and predicts real estate prices using an ensemble regressor. The app supports dynamic datasets via CSV upload or local path input.

## Features
- Interactive EDA with Plotly visualizations
- End-to-end ML pipeline (cleaning, feature selection, training, evaluation)
- Ensemble modeling (Random Forest, Gradient Boosting, AdaBoost, etc.)
- Price predictions with currency conversion
- Dynamic dataset loading (upload or local path)

## Tech Stack
- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Plotly, Matplotlib, Seaborn

## Project Structure
```
ANN_CA/
  app.py
  real_estate_pipeline.py
  cleaned_data.csv
  ensemble_model.pkl
  model.pkl
  scaler.pkl
  feature_cols.pkl
```

## Getting Started
1. Create and activate a virtual environment
2. Install dependencies
3. Run the app

```
pip install -r requirements.txt
streamlit run app.py
```

## Using Your Own Dataset
1. Open the app
2. In the sidebar under **Dataset**, upload a CSV or provide a local path
3. Click **Initialize ML Pipeline**

The app will adapt and train on the chosen dataset.

## Model Training
Go to **Model Training** and click **Train All Models**. Trained artifacts are saved in the project folder:
- `ensemble_model.pkl`
- `model.pkl`
- `scaler.pkl`
- `feature_cols.pkl`

## Notes
- Default dataset is `cleaned_data.csv`
- The pipeline auto-detects a target column from common names like `Price`

## Team
- Ayush Pandey
- Chirag Pandey
