import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
from pathlib import Path
from typing import Dict, Union
import numpy as np
import pandas as pd
from config import MODEL_FILE, RANDOM_STATE, N_ESTIMATORS

# Configure logger
logger = logging.getLogger(__name__)


class HeartDiseaseModel:
    """
    Wrapper for the Heart Disease Prediction Model (Random Forest).
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
        )

    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> None:
        logger.info("Training model...")
        self.model.fit(X_train, y_train)
        logger.info("Model trained.")

    def evaluate(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Union[float, str, np.ndarray]]:
        logger.info("Evaluating model...")
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info("\nClassification Report:\n" + report)
        logger.info("\nConfusion Matrix:\n" + str(cm))

        return {"accuracy": acc, "report": report, "confusion_matrix": cm}

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, filepath: Path = MODEL_FILE) -> None:
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: Path = MODEL_FILE) -> "HeartDiseaseModel":
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self
