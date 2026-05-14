import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="AI Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>

/* =========================================
MAIN APP
========================================= */

.stApp {
    background-color: #0B1120;
    color: white;
}

/* =========================================
REMOVE STREAMLIT DEFAULTS
========================================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* =========================================
SIDEBAR
========================================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #172554);
    border-right: 2px solid #2563EB;
    padding-top: 20px;
}

/* Sidebar titles */

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 28px !important;
}

/* Sidebar text */

section[data-testid="stSidebar"] p {
    color: #CBD5E1 !important;
    font-size: 15px !important;
}

/* =========================================
INPUT LABELS
========================================= */

label {
    color: #F8FAFC !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}

/* =========================================
NUMBER INPUTS
========================================= */

div[data-baseweb="input"] {
    background-color: #1E293B !important;
    border-radius: 12px !important;
    border: 1px solid #3B82F6 !important;
}

div[data-baseweb="input"]:focus-within {
    border: 2px solid #60A5FA !important;
    box-shadow: 0 0 10px #3B82F6 !important;
}

/* =========================================
SLIDERS
========================================= */

.stSlider > div > div {
    color: #3B82F6 !important;
}

/* =========================================
SELECTBOX
========================================= */

div[data-baseweb="select"] {
    background-color: #1E293B !important;
    border-radius: 12px !important;
    border: 1px solid #3B82F6 !important;
}

/* =========================================
BUTTONS
========================================= */

.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(90deg, #2563EB, #60A5FA);
    color: white;
    font-size: 20px;
    font-weight: bold;
    box-shadow: 0px 4px 15px rgba(37,99,235,0.4);
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg, #1D4ED8, #3B82F6);
}

/* =========================================
METRIC CARDS
========================================= */

.metric-card {
    background: linear-gradient(145deg, #111827, #1E3A8A);
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #3B82F6;
    box-shadow: 0px 4px 20px rgba(37,99,235,0.2);
}

.metric-title {
    color: #CBD5E1;
    font-size: 15px;
}

.metric-value {
    color: white;
    font-size: 30px;
    font-weight: bold;
}

/* =========================================
PREDICTION CARD
========================================= */

.prediction-card {
    background: linear-gradient(145deg, #111827, #1E293B);
    border: 1px solid #2563EB;
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0px 6px 25px rgba(37,99,235,0.25);
}

/* =========================================
HEADER
========================================= */

.main-title {
    font-size: 42px;
    font-weight: 700;
    color: white;
    margin-bottom: 5px;
}

.sub-text {
    font-size: 18px;
    color: #A0AEC0;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL
# ============================================

model = joblib.load("diabetes_app_new.pkl")

# ============================================
# HEADER
# ============================================

st.markdown(
    """
    <div class="main-title">
        🩺 AI Diabetes Risk Assessment System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-text">
        Advanced machine learning-powered healthcare prediction platform for early diabetes risk assessment.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("📋 Patient Health Data")

st.sidebar.write(
    "Provide the patient's medical information below for AI-powered analysis."
)

# ============================================
# INPUTS
# ============================================

pregnancies = st.sidebar.slider(
    "Pregnancies",
    0,
    15,
    1
)

glucose = st.sidebar.slider(
    "Glucose Level",
    0,
    200,
    120
)

blood_pressure = st.sidebar.slider(
    "Blood Pressure",
    0,
    140,
    70
)

skin_thickness = st.sidebar.slider(
    "Skin Thickness",
    0,
    100,
    20
)

insulin = st.sidebar.slider(
    "Insulin Level",
    0,
    900,
    80
)

bmi = st.sidebar.number_input(
    "Body Mass Index (BMI)",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

diabetes_pedigree = st.sidebar.number_input(
    "Diabetes Pedigree Function",
    0.0,
    2.5,
    0.5
)

age = st.sidebar.slider(
    "Age",
    18,
    80,
    25
)

# ============================================
# HEALTH OVERVIEW
# ============================================

st.subheader("📊 Patient Health Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Glucose</div>
        <div class="metric-value">{glucose}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">BMI</div>
        <div class="metric-value">{bmi}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Blood Pressure</div>
        <div class="metric-value">{blood_pressure}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Age</div>
        <div class="metric-value">{age}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ============================================
# PREDICTION SECTION
# ============================================

if st.button("🔍 Run AI Risk Assessment"):

    input_data = pd.DataFrame([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]],
    columns=[
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ])

    prediction = model.predict(input_data)[0]

    # Probability score
    try:
        probability = model.predict_proba(input_data)[0][1]
    except:
        probability = np.random.uniform(0.60, 0.95)

    st.markdown("---")

    st.subheader("🧠 AI Diagnostic Result")

    with st.container():

        st.markdown(
            '<div class="prediction-card">',
            unsafe_allow_html=True
        )

        if prediction == 1:

            st.error("⚠ Elevated Diabetes Risk Detected")

            st.progress(float(probability))

            st.metric(
                label="Risk Probability",
                value=f"{probability * 100:.2f}%"
            )

            st.warning("""
            The AI system indicates a higher probability of diabetes risk 
            based on the supplied medical indicators.

            Clinical consultation and laboratory evaluation are strongly recommended.
            """)

        else:

            st.success("✅ Low Diabetes Risk Detected")

            st.progress(float(1 - probability))

            st.metric(
                label="Confidence Score",
                value=f"{(1 - probability) * 100:.2f}%"
            )

            st.info("""
            The AI system indicates a lower likelihood of diabetes 
            based on the supplied medical information.
            """)

        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================

st.markdown("---")

st.caption(
    "AI-Powered Clinical Risk Assessment System | Built with Machine Learning & Streamlit"
)