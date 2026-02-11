import pytest
import pandas as pd
import numpy as np
from preprocess import FeatureEngineer, HeartDiseasePreprocessor


@pytest.fixture
def sample_data():
    """Creates a sample DataFrame for testing."""
    data = {
        "age": [50, 60, 40],
        "sex": [1, 0, 1],
        "trestbps": [120, 150, 130],
        "chol": [200, 250, 220],
        "thalch": [150, 140, 160],
        "oldpeak": [1.0, 3.0, 0.0],
        "ca": [0, 2, 1],
        "fbs": ["TRUE", "FALSE", "True"],
        "exang": ["FALSE", "TRUE", "False"],
        "cp": [1, 2, 3],
        "restecg": [0, 1, 0],
        "slope": [1, 2, 1],
        "thal": [3, 7, 3],
    }
    return pd.DataFrame(data)


def test_feature_engineer_transform(sample_data):
    """Test FeatureEngineer transformation logic."""
    fe = FeatureEngineer()
    transformed_data = fe.fit_transform(sample_data)

    # Check if new features are created
    assert "chol_age_ratio" in transformed_data.columns
    assert "bp_above_140" in transformed_data.columns
    assert "oldpeak_high" in transformed_data.columns
    assert "stress_reserve" in transformed_data.columns
    assert "ca_binary" in transformed_data.columns

    # Check specific calculations
    assert transformed_data["bp_above_140"].iloc[0] == 0  # 120 <= 140
    assert transformed_data["bp_above_140"].iloc[1] == 1  # 150 > 140

    # Check boolean string conversion
    assert transformed_data["fbs"].dtype == float
    assert transformed_data["fbs"].iloc[0] == 1.0


def test_preprocessor_pipeline(sample_data):
    """Test the full HeartDiseasePreprocessor pipeline."""
    preprocessor = HeartDiseasePreprocessor()

    # Fit and transform
    X_processed = preprocessor.fit_transform(sample_data)

    # Check output shape
    # We have numeric features (original + new) + categorical features (one-hot encoded)
    # Exact number depends on one-hot encoding categories, but should be numpy array
    assert isinstance(X_processed, np.ndarray)
    assert X_processed.shape[0] == 3  # same number of rows
    assert X_processed.shape[1] > 0  # some columns


def test_preprocessor_save_load(tmp_path):
    """Test saving and loading the preprocessor."""
    preprocessor = HeartDiseasePreprocessor()
    filepath = tmp_path / "preprocessor.joblib"

    preprocessor.save(filepath)
    assert filepath.exists()

    loaded_preprocessor = HeartDiseasePreprocessor()
    loaded_preprocessor.load(filepath)
    assert loaded_preprocessor.pipeline is not None
