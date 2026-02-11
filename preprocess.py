import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import logging
from pathlib import Path

from typing import Optional
from config import PREPROCESSOR_FILE, NUMERIC_FEATURES, CATEGORICAL_FEATURES

# Configure logger
logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering."""

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Ensure numerical columns are numeric
        for col in ["chol", "age", "trestbps", "oldpeak", "thalch", "ca"]:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        # Create new features if columns exist
        if "chol" in X.columns and "age" in X.columns:
            X["chol_age_ratio"] = X["chol"] / X["age"]
            X.replace([np.inf, -np.inf], np.nan, inplace=True)

        # boolean string conversion
        for col in ["fbs", "exang"]:
            if col in X.columns:
                X[col] = (
                    X[col]
                    .replace(
                        {
                            "TRUE": 1,
                            "FALSE": 0,
                            "True": 1,
                            "False": 0,
                            True: 1,
                            False: 0,
                        }
                    )
                    .astype(float)
                )

        if "trestbps" in X.columns:
            X["bp_above_140"] = (X["trestbps"] > 140).astype(int)

        if "oldpeak" in X.columns:
            X["oldpeak_high"] = (X["oldpeak"] > 2).astype(int)

        if "thalch" in X.columns and "age" in X.columns:
            X["stress_reserve"] = X["thalch"] - X["age"]

        if "ca" in X.columns:
            X["ca_binary"] = (X["ca"] > 0).astype(int)

        return X


class HeartDiseasePreprocessor:
    """
    Wrapper class for the preprocessing pipeline.
    """

    def __init__(self):
        self.pipeline = None
        self._build_pipeline()

    def _build_pipeline(self):
        # Numerical pipeline: Median imputation -> Scaling
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # Categorical pipeline: Mode imputation -> OneHotEncoder
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        # Feature engineering wrapper
        # We apply feature engineering FIRST, then the column transformer
        # But wait, ColumnTransformer selects columns. Newly created features won't be in the list defined in config.
        # So we need to handle this.

        # Revised approach:
        # 1. Feature Engineering (pandas in -> pandas out) using custom transformer
        # 2. Imputation & Scaling/Encoding (pandas in -> pandas/numpy out)

        # Since sklearn < 1.4 doesn't output pandas by default easily, we will manage column names carefully or use the pipeline end-to-end.

        self.feature_engineer = FeatureEngineer()

        # We need to know the 'new' numeric features added by FeatureEngineer to include them in scaling
        self.new_numeric_features = [
            "chol_age_ratio",
            "bp_above_140",
            "oldpeak_high",
            "stress_reserve",
            "ca_binary",
        ]
        self.all_numeric_features = NUMERIC_FEATURES + self.new_numeric_features

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.all_numeric_features),
                ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ],
            remainder="drop",  # Drop other columns (like original 'num' target if passed, or unused ones)
        )

        self.pipeline = Pipeline(
            steps=[
                ("engineered", self.feature_engineer),
                ("preprocessor", self.preprocessor),
            ]
        )

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "HeartDiseasePreprocessor":
        self.pipeline.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.transform(X)

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> np.ndarray:
        return self.pipeline.fit_transform(X, y)

    def save(self, filepath: Path = PREPROCESSOR_FILE) -> None:
        joblib.dump(self.pipeline, filepath)
        logger.info(f"Preprocessor saved to {filepath}")

    def load(self, filepath: Path = PREPROCESSOR_FILE) -> "HeartDiseasePreprocessor":
        self.pipeline = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")
        return self
