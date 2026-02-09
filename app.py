import streamlit as st
import pandas as pd
import joblib
from config import MODEL_FILE, PREPROCESSOR_FILE

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_FILE)
        preprocessor = joblib.load(PREPROCESSOR_FILE)
        return model, preprocessor
    except FileNotFoundError:
        st.error("Model artifacts not found. Please run train.py first.")
        return None, None

def main():
    st.title("❤️ Heart Disease Risk Predictor")
    st.write("Enter patient details to estimate the risk of heart disease.")

    model, preprocessor = load_artifacts()
    
    if not model or not preprocessor:
        return

    # Input Form
    with st.form("prediction_form"):
        st.header("Patient Information")
        
        c1, c2 = st.columns(2)
        
        with c1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", [
                "typical angina", 
                "atypical angina", 
                "non-anginal", 
                "asymptomatic"
            ])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
            chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["FALSE", "TRUE"])
        
        with c2:
            restecg = st.selectbox("Resting ECG Results", [
                "normal", 
                "lv hypertrophy", 
                "st-t abnormality"
            ])
            thalch = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
            exang = st.selectbox("Exercise Induced Angina", ["FALSE", "TRUE"])
            oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", [
                "upsloping", 
                "flat", 
                "downsloping"
            ])
            ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=4, value=0) # Data allows 0-3, sometimes 4 in dirty data but 3 is major
            thal = st.selectbox("Thalassemia", [
                "normal", 
                "fixed defect", 
                "reversable defect"
            ])

        submit_btn = st.form_submit_button("Predict Risk")

    if submit_btn:
        # Create DataFrame
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'cp': [cp],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalch': [thalch],
            'exang': [exang],
            'oldpeak': [oldpeak],
            'slope': [slope],
            'ca': [ca],
            'thal': [thal]
        })
        
        # Preprocess
        try:
            processed_data = preprocessor.transform(input_data)
            
            # Predict
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0][1]
            
            st.divider()
            st.subheader("Prediction Result")
            
            if prediction == 1:
                st.error(f"⚠️ High Risk of Heart Disease detected.")
                st.write(f"Confidence: **{probability:.2%}**")
            else:
                st.success(f"✅ Low Risk of Heart Disease detected.")
                st.write(f"Confidence: **{(1-probability):.2%}**") # Prob of class 0
                
            with st.expander("Show Details"):
                st.json(input_data.to_dict(orient='records')[0])
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
