import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config import MODEL_FILE, RANDOM_STATE, N_ESTIMATORS

class HeartDiseaseModel:
    """
    Wrapper for the Heart Disease Prediction Model (Random Forest).
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, 
            random_state=RANDOM_STATE
        )

    def train(self, X_train, y_train):
        print("Training model...")
        self.model.fit(X_train, y_train)
        print("Model trained.")

    def evaluate(self, X_test, y_test):
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:\n")
        print(report)
        print("\nConfusion Matrix:\n")
        print(cm)
        
        return {
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm
        }

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, filepath=MODEL_FILE):
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath=MODEL_FILE):
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self
