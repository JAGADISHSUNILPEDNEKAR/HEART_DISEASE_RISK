import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_FILE, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
from preprocess import HeartDiseasePreprocessor
import logging
import sys
from model import HeartDiseaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    # 1. Load Data
    logger.info(f"Loading data from {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        logger.error(f"Error: Data file not found at {DATA_FILE}")
        return

    # Basic cleaning corresponding to "Convert boolean to int" from original notebook if needed,
    # but the FeatureEngineer handles most.
    # We should ensure target is binary 1/0 as per notebook: df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

    logger.info("Preparing data...")
    if TARGET_COLUMN in df.columns:
        # The dataset 'num' column is 0 (no disease) and 1,2,3,4 (disease degrees)
        # We convert to binary classification
        df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda x: 1 if x > 0 else 0)

        X = df.drop(columns=[TARGET_COLUMN, "id", "dataset"], errors="ignore")
        y = df[TARGET_COLUMN]
    else:
        logger.error(f"Error: Target column '{TARGET_COLUMN}' not found.")
        return

    # 2. Split Data
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 3. Preprocessing
    logger.info("Preprocessing data...")
    preprocessor = HeartDiseasePreprocessor()

    # Fit on train, transform both
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save preprocessor
    preprocessor.save()

    # 4. Train Model
    model = HeartDiseaseModel()
    model.train(X_train_processed, y_train)

    # 5. Evaluate
    model.evaluate(X_test_processed, y_test)

    # 6. Save Model
    model.save()

    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
