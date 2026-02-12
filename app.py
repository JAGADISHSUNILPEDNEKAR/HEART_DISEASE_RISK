import streamlit as st
import pandas as pd
import joblib
import logging
import sys
from config import MODEL_FILE, PREPROCESSOR_FILE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("assets/style.css")


@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_FILE)
        preprocessor = joblib.load(PREPROCESSOR_FILE)
        return model, preprocessor
    except FileNotFoundError:
        st.error("Model artifacts not found. Please run train.py first.")
        logger.error("Model artifacts not found.")
        return None, None


def main():
    # Sidebar
    with st.sidebar:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/2503/2503639.png", width=100
        )  # Example icon
        st.title("HeartGuard AI")
        st.info(
            """
        **About**
        This application uses Machine Learning to estimate the risk of heart disease based on clinical parameters.
        """
        )
        st.divider()
        st.write("Built with ❤️ using Streamlit")

    # Main Content
    st.title("🫀 Heart Disease Risk Predictor")
    st.markdown(
        "### Enter patient details below to generate a real-time risk assessment."
    )

    model, preprocessor = load_artifacts()

    if not model or not preprocessor:
        return

    # Input Form
    with st.form("prediction_form"):
        st.markdown("#### 📋 Patient Information")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Demographics**")
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", ["Male", "Female"])

        with c2:
            st.markdown("**Vitals**")
            trestbps = st.number_input(
                "Resting BP (mm Hg)", min_value=50, max_value=250, value=120
            )
            chol = st.number_input(
                "Cholesterol (mg/dl)", min_value=100, max_value=600, value=200
            )

        with c3:
            st.markdown("**History**")
            fbs = st.selectbox("Fasting BS > 120 mg/dl", ["FALSE", "TRUE"])
            exang = st.selectbox("Exercise Induced Angina", ["FALSE", "TRUE"])

        st.divider()
        st.markdown("#### 🩺 Clinical Metrics")

        c4, c5 = st.columns(2)
        with c4:
            cp = st.selectbox(
                "Chest Pain Type",
                ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
            )
            restecg = st.selectbox(
                "Resting ECG Results", ["normal", "lv hypertrophy", "st-t abnormality"]
            )
            thalch = st.number_input(
                "Max Heart Rate", min_value=50, max_value=250, value=150
            )

        with c5:
            oldpeak = st.number_input(
                "ST Depression (Exercise)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
            )
            slope = st.selectbox(
                "Slope of Peak ST",
                ["upsloping", "flat", "downsloping"],
            )
            ca = st.number_input(
                "Major Vessels (0-3)", min_value=0, max_value=4, value=0
            )
            thal = st.selectbox(
                "Thalassemia", ["normal", "fixed defect", "reversable defect"]
            )

        st.divider()
        submit_btn = st.form_submit_button("Generate Risk Assessment")

    if submit_btn:
        # Create DataFrame
        input_data = pd.DataFrame(
            {
                "age": [age],
                "sex": [sex],
                "cp": [cp],
                "trestbps": [trestbps],
                "chol": [chol],
                "fbs": [fbs],
                "restecg": [restecg],
                "thalch": [thalch],
                "exang": [exang],
                "oldpeak": [oldpeak],
                "slope": [slope],
                "ca": [ca],
                "thal": [thal],
            }
        )

        # Preprocess
        try:
            processed_data = preprocessor.transform(input_data)

            # Predict
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0][1]

            st.divider()
            st.subheader("📊 Assessment Result")

            c_res1, c_res2 = st.columns([1, 2])

            with c_res1:
                if prediction == 1:
                    st.metric(
                        label="Risk Level",
                        value="HIGH",
                        delta="Attention Needed",
                        delta_color="inverse",
                    )
                else:
                    st.metric(
                        label="Risk Level",
                        value="LOW",
                        delta="Normal",
                        delta_color="normal",
                    )

            with c_res2:
                st.write("Is there a risk of heart disease?")
                risk_percent = probability if prediction == 1 else (1 - probability)
                
                # Custom progress bar color
                bar_color = "#ff4b4b" if prediction == 1 else "#00c853"
                st.markdown(
                    f"""
                    <div style="background-color: #444; border-radius: 10px; padding: 3px;">
                        <div style="width: {risk_percent*100}%; background-color: {bar_color}; height: 20px; border-radius: 7px;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(f"Confidence: {risk_percent:.2%}")

            if prediction == 1:
                st.error(
                    "The model predicts a high risk of heart disease. Please consult a cardiologist for further evaluation."
                )
            else:
                st.success(
                    "The model predicts a low risk of heart disease. Maintain a healthy lifestyle!"
                )

            with st.expander("Show detailed input data"):
                st.json(input_data.to_dict(orient="records")[0])

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            logger.error(f"Error during prediction: {e}", exc_info=True)


if __name__ == "__main__":
    main()
