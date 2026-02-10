import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from model import HeartDiseaseModel

@pytest.fixture
def mock_data():
    """Creates mock X and y data."""
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    return X, y

def test_model_initialization():
    """Test if the model initializes with correct parameters."""
    model = HeartDiseaseModel()
    assert model.model is not None

def test_model_train(mock_data):
    """Test model training."""
    X, y = mock_data
    model = HeartDiseaseModel()
    
    # Train the model
    model.train(X, y)
    
    # Ensure it's fitted (sklearn models usually have attributes ending in _ after fit)
    # Or simply check if we can predict without error
    try:
        model.predict(X)
    except Exception as e:
        pytest.fail(f"Model prediction failed after training: {e}")

def test_model_evaluate(mock_data):
    """Test model evaluation."""
    X, y = mock_data
    model = HeartDiseaseModel()
    model.train(X, y)
    
    metrics = model.evaluate(X, y)
    
    assert "accuracy" in metrics
    assert "report" in metrics
    assert "confusion_matrix" in metrics
    assert isinstance(metrics["accuracy"], float)

def test_model_save_load(tmp_path, mock_data):
    """Test saving and loading the model."""
    X, y = mock_data
    model = HeartDiseaseModel()
    model.train(X, y)
    
    filepath = tmp_path / "model.joblib"
    model.save(filepath)
    assert filepath.exists()
    
    loaded_model = HeartDiseaseModel()
    loaded_model.load(filepath)
    
    # Check if loaded model can predict
    assert loaded_model.model is not None
