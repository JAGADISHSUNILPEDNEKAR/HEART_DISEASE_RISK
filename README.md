# Heart Disease Prediction

A production-ready Machine Learning project structure for heart disease prediction, deployed using Streamlit.

**Google Collab link** - https://colab.research.google.com/drive/1TvHN_NCEfuiHN4rsXoSn-Pw44BIrQLT8#scrollTo=M0P7uDbhPX9Y

## Structure

```
HEART_DISEASE_RISK/
├── app.py                # Streamlit application
├── train.py              # Training pipeline script
├── preprocess.py         # Data preprocessing module
├── model.py              # Model definition module
├── config.py             # Configuration settings
├── requirements.txt      # Project dependencies
├── data/                 # Data directory (contains heart_disease_uci.csv)
└── models/               # Saved models directory
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd HEART_DISEASE_RISK
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

Run the training pipeline to process data, train the model, and save artifacts to `models/`.

```bash
python train.py
```

### Running the App

Launch the Streamlit application to predict heart disease risk interactively.

```bash
streamlit run app.py
```

## Deployment

To deploy on Streamlit Cloud:

1.  Push your code to GitHub.
2.  Log in to [Streamlit Cloud](https://streamlit.io/cloud).
3.  Connect your GitHub account and select this repository.
4.  Set the main file path to `app.py`.
5.  Click **Deploy**.

## Model Details

-   **Algorithm:** Random Forest Classifier
-   **Preprocessing:** Median imputation for numerical features, Mode imputation for categorical features, One-Hot Encoding and Scaling.
-   **Metrics:** Accuracy, Classification Report
