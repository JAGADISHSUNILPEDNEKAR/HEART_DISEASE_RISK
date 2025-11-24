# Heart Disease Risk Prediction: Logistic Regression vs Random Forest

**Student:** Jagadish Sunil Pednekar (Roll No: 240410700005)  
**Project Type:** Data Scientist OJT Project  
**Reference:** OJT Project PRD Sem - 3

## Overview

This project predicts heart disease risk using the UCI Heart Disease dataset by comparing two machine learning models: Logistic Regression and Random Forest. The implementation follows the PRD specifications with modular code, reproducible experiments (random_state=42), comprehensive evaluation metrics, and a Streamlit demo dashboard.

## Features

- ✅ Data preprocessing with scaling and outlier handling
- ✅ Logistic Regression and Random Forest classifiers
- ✅ K-fold cross-validation (k=5)
- ✅ Threshold tuning for optimal precision-recall balance
- ✅ Comprehensive metrics: accuracy, precision, recall, F1, ROC-AUC
- ✅ Feature importance and coefficient visualization
- ✅ Calibration curve analysis
- ✅ Streamlit web dashboard for real-time predictions
- ✅ Unit tests and CI pipeline

## Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd HEART_DISEASE_RISK
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download the UCI Heart Disease dataset and place it as `data/heart.csv`. The dataset should contain these features:
- age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target

Dataset source: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)

## Usage

### Train Models
```bash
python scripts/train_models.py
```

This will:
- Load and preprocess data from `data/heart.csv`
- Train Logistic Regression and Random Forest models
- Save models to `models/` directory
- Perform 5-fold cross-validation
- Print initial performance metrics

### Evaluate Models
```bash
python scripts/evaluate_models.py
```

Generates:
- Performance comparison table
- ROC curves
- Confusion matrices
- Feature importance plots
- Calibration curves
- Saved visualizations in project root

### Hyperparameter Tuning
```bash
python scripts/tune_hyperparameters.py
```

Performs grid search for Random Forest hyperparameters (n_estimators, max_depth, min_samples_split).

### Run Streamlit Dashboard
```bash
streamlit run app.py
```

Interactive web app for:
- Single patient risk prediction
- Model performance comparison
- Feature importance visualization

### Run Tests
```bash
pytest tests/ -v
```

### Exploratory Data Analysis

Open `notebooks/01_exploratory_data_analysis.ipynb` in Jupyter to explore:
- Data distributions
- Correlation analysis
- Missing value checks
- Class balance analysis

## Project Structure
```
src/
├── config.py              # Configuration and constants
├── data_preprocessing.py  # Data loading and preprocessing pipeline
├── model_trainer.py       # Model training with cross-validation
├── model_evaluator.py     # Evaluation metrics and visualization
└── utils.py               # Utility functions

scripts/
├── train_models.py        # Training script
├── evaluate_models.py     # Evaluation script
└── tune_hyperparameters.py # Hyperparameter tuning

tests/
├── test_data_preprocessing.py
└── test_model_trainer.py
```

## Technical Decisions

1. **Random State**: Set to 42 throughout for reproducibility
2. **Scaling**: StandardScaler applied to all numerical features
3. **Cross-Validation**: 5-fold stratified CV for robust evaluation
4. **Threshold Tuning**: Default threshold adjusted for precision-recall balance
5. **Random Forest Defaults**: n_estimators=100, max_depth=10, min_samples_split=10

## Success Metrics (from PRD)

Target: ≥85% accuracy with balanced precision-recall trade-off

## Next Improvements

1. Handle class imbalance with SMOTE or class weights
2. Add more ensemble methods (XGBoost, LightGBM)
3. Implement SHAP for model interpretability
4. Deploy to cloud platform (Heroku, AWS)
5. Add model monitoring and drift detection
6. Expand calibration analysis
7. Implement automated retraining pipeline

## License

Project for OJT Semester 3.