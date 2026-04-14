"""
Real Estate Price Prediction Pipeline
Lightweight, self-contained pipeline for the Streamlit app.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


class RealEstatePricePrediction:
    def __init__(self, data_path: str = "cleaned_data.csv") -> None:
        self.data_path = data_path
        self.raw_df: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None

        self.target_col: Optional[str] = None
        self.feature_cols: List[str] = []

        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        self.models: Dict[str, object] = {}
        self.performance: Dict[str, Dict[str, float]] = {}
        self.cv_results: Dict[str, Dict[str, object]] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None

        self.scaler = StandardScaler()

    def load_data(self, data_path: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> None:
        if df is not None:
            self.raw_df = df.copy()
            self.df = df.copy()
            return

        path = data_path or self.data_path
        if not os.path.isabs(path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.abspath(os.path.join(base_dir, path))
        self.raw_df = pd.read_csv(path)
        self.df = self.raw_df.copy()

    def perform_eda(self) -> None:
        if self.df is None:
            raise ValueError("Dataframe not loaded.")
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            self.correlation_matrix = pd.DataFrame()
        else:
            self.correlation_matrix = numeric_df.corr()

    def clean_and_engineer(self) -> None:
        if self.df is None:
            raise ValueError("Dataframe not loaded.")
        df = self.df.copy()

        # Basic numeric cleanup
        for col in df.columns:
            if df[col].dtype == object:
                # try numeric conversion where possible
                df[col] = pd.to_numeric(df[col], errors="ignore")

        # Fill missing values
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

        # Example engineered features (safe when columns exist)
        if "Bedrooms" in df.columns and "Bathrooms" in df.columns:
            df["Room_Total"] = df["Bedrooms"] + df["Bathrooms"]
        if "Location_Score" in df.columns:
            df["Luxury_Score"] = df["Location_Score"].astype(float)

        self.df = df

    def _infer_target(self) -> str:
        if self.df is None:
            raise ValueError("Dataframe not loaded.")
        for col in ["Price", "price", "SalePrice", "Target"]:
            if col in self.df.columns:
                return col
        # fallback: last numeric column
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric target column found.")
        return numeric_cols[-1]

    def select_features(self) -> None:
        if self.df is None:
            raise ValueError("Dataframe not loaded.")

        self.target_col = self._infer_target()
        df = self.df.copy()

        y = pd.to_numeric(df[self.target_col], errors="coerce")
        df = df.drop(columns=[self.target_col])

        # One-hot encode categoricals
        df = pd.get_dummies(df, drop_first=False)

        # Align with numeric target rows
        mask = y.notna()
        self.y = y[mask]
        self.X = df.loc[mask].reset_index(drop=True)
        self.y = self.y.reset_index(drop=True)

        self.feature_cols = self.X.columns.tolist()

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        if self.X is None or self.y is None:
            raise ValueError("Features not selected.")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def initialize_models(self) -> None:
        self.models = {
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "AdaBoost": AdaBoostRegressor(random_state=42),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.01, max_iter=5000),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
        }

    def train_models(self) -> None:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not prepared.")

        X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.trained_models = {}
        for name, model in self.models.items():
            model.fit(X_train_scaled, self.y_train)
            self.trained_models[name] = model

    def perform_kfold_validation(self, n_splits: int = 5) -> None:
        if self.X is None or self.y is None:
            raise ValueError("Features not selected.")
        X_scaled = self.scaler.fit_transform(self.X)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.cv_results = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, X_scaled, self.y, cv=kf, scoring="r2")
            self.cv_results[name] = {"mean": float(scores.mean()), "std": float(scores.std()), "scores": scores}

    def evaluate_models(self) -> None:
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not prepared.")
        X_test_scaled = self.scaler.transform(self.X_test)
        self.performance = {}
        for name, model in self.trained_models.items():
            preds = model.predict(X_test_scaled)
            mse = mean_squared_error(self.y_test, preds)
            self.performance[name] = {
                "R2": float(r2_score(self.y_test, preds)),
                "RMSE": float(np.sqrt(mse)),
                "MAE": float(mean_absolute_error(self.y_test, preds)),
            }

    def save_models(self) -> None:
        if not getattr(self, "trained_models", None):
            raise ValueError("Models not trained.")

        # Voting ensemble of three strong defaults
        estimators = [
            ("rf", self.trained_models.get("Random Forest")),
            ("gbr", self.trained_models.get("Gradient Boosting")),
            ("ada", self.trained_models.get("AdaBoost")),
        ]
        estimators = [e for e in estimators if e[1] is not None]

        ensemble = VotingRegressor(estimators=estimators)
        ensemble.fit(self.scaler.transform(self.X_train), self.y_train)

        joblib.dump(ensemble, "ensemble_model.pkl")
        joblib.dump(self.trained_models.get("Random Forest"), "model.pkl")
        joblib.dump(self.scaler, "scaler.pkl")
        joblib.dump(self.feature_cols, "feature_cols.pkl")

    def get_feature_importance(self) -> pd.DataFrame:
        model = self.trained_models.get("Random Forest") if getattr(self, "trained_models", None) else None
        if model is None:
            return pd.DataFrame({"Feature": self.feature_cols, "Importance": np.zeros(len(self.feature_cols))})
        importances = getattr(model, "feature_importances_", np.zeros(len(self.feature_cols)))
        return pd.DataFrame({"Feature": self.feature_cols, "Importance": importances}).sort_values(
            "Importance", ascending=False
        )
