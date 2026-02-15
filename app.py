import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
)

# ──────────────────────────────────────────────
# Custom CSS for a premium look
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #888;
        font-size: 1.05rem;
    }

    .result-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .churn-yes {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
    }
    .churn-no {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .result-card h2 {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .result-card p {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.3rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #667eea22;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Load the pre-trained model from .pkl file
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('churn_prediction_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 Customer Churn Predictor</h1>
    <p>Enter customer details below to predict if they are likely to churn</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ──────────────────────────────────────────────
# Input Form
# ──────────────────────────────────────────────

# --- Account Information ---
st.markdown('<p class="section-header">🏦 Account Information</p>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
with col2:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=50.0, step=0.5)
with col3:
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=600.0, step=10.0)

st.markdown("")

# --- Personal Information ---
st.markdown('<p class="section-header">👤 Personal Information</p>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
with col3:
    partner = st.selectbox("Partner", ["No", "Yes"])
with col4:
    dependents = st.selectbox("Dependents", ["No", "Yes"])

st.markdown("")

# --- Phone Service ---
st.markdown('<p class="section-header">📞 Phone Service</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
with col2:
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

st.markdown("")

# --- Internet Service ---
st.markdown('<p class="section-header">🌐 Internet Service</p>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
with col2:
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
with col3:
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

col1, col2, col3 = st.columns(3)
with col1:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
with col2:
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
with col3:
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])

col1, _ = st.columns(2)
with col1:
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.markdown("")

# --- Billing & Contract ---
st.markdown('<p class="section-header">💳 Billing & Contract</p>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
with col2:
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
with col3:
    payment_method = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ])

st.divider()

# ──────────────────────────────────────────────
# Predict Button
# ──────────────────────────────────────────────
senior_citizen_val = 1 if senior_citizen == "Yes" else 0

if st.button("Predict Churn"):
    # Build input DataFrame matching the training features
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'gender': [gender],
        'SeniorCitizen': [senior_citizen_val],
        'Partner': [partner],
        'Dependents': [dependents],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
    })

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    churn_prob = probability[1] * 100
    no_churn_prob = probability[0] * 100

    # Display result
    if prediction == 1:
        st.markdown(f"""
        <div class="result-card churn-yes">
            <h2>⚠️ High Churn Risk</h2>
            <p>This customer is <strong>likely to churn</strong> with a probability of <strong>{churn_prob:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card churn-no">
            <h2>✅ Low Churn Risk</h2>
            <p>This customer is <strong>likely to stay</strong> with a probability of <strong>{no_churn_prob:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # Show probability breakdown
    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Churn Probability", value=f"{churn_prob:.1f}%")
    with col2:
        st.metric(label="Retention Probability", value=f"{no_churn_prob:.1f}%")

    # Show input summary in an expander
    with st.expander("📋 View Input Summary"):
        st.dataframe(input_data.T.rename(columns={0: 'Value'}), use_container_width=True)
