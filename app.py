import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt

# Page configuration (must be first Streamlit command)
st.set_page_config(
    layout="wide",
    page_title="CardioCheck",
    page_icon="❤️"
)

# Custom CSS injection
custom_css = """
<style>
/* Main app background */
.stApp {
    background-color: #FFF5F5;
}

/* Button styling */
.stButton>button {
    background-color: #DC143C;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background-color: #FF6B6B;
    box-shadow: 0 4px 12px rgba(220, 20, 60, 0.3);
    transform: translateY(-2px);
}

/* Form containers */
.css-1d391kg, .css-12oz5g7 {
    background-color: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Input fields */
.stTextInput>div>div>input {
    border-radius: 8px;
    border: 2px solid #E2E8F0;
    padding: 0.75rem;
    transition: border-color 0.3s ease;
}

.stTextInput>div>div>input:focus {
    border-color: #DC143C;
    box-shadow: 0 0 0 3px rgba(220, 20, 60, 0.1);
}

/* Selectbox styling */
.stSelectbox>div>div {
    border-radius: 8px;
}

/* Number input styling */
.stNumberInput>div>div>input {
    border-radius: 8px;
    border: 2px solid #E2E8F0;
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #FFFFFF;
}

/* Headers */
h1 {
    color: #DC143C;
    font-weight: 700;
}

h2, h3 {
    color: #2D3748;
}

/* Metric styling */
[data-testid="stMetricValue"] {
    font-size: 2rem;
    font-weight: 700;
    color: #DC143C;
}

/* Cards and containers */
.element-container {
    background-color: white;
    padding: 1rem;
    border-radius: 8px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #DC143C;
    color: white;
}

/* Expander styling */
.streamlit-expanderHeader {
    background-color: #FFF5F5;
    border-radius: 8px;
    font-weight: 600;
}

/* Form styling */
.stForm {
    background-color: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Firebase Admin SDK initialization
def init_firebase():
    """Initialize Firebase Admin SDK if not already initialized"""
    if not firebase_admin._apps:
        firebase_admin.initialize_app()

# Initialize Firebase
init_firebase()

# Train model function
@st.cache_resource
def train_model():
    """
    Train RandomForest model on heart disease dataset with SHAP explainer
    Returns: (pipeline, shap_explainer, dataframe)
    """
    try:
        # Load dataset from URL
        url = "https://storage.googleapis.com/applied-dl/heart-disease-dataset/heart.csv"
        df = pd.read_csv(url)

        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']

        # Define feature types
        categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        # Create full pipeline with RandomForest
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # Train on entire dataset
        pipeline.fit(X, y)

        # Initialize SHAP explainer with sample background
        background_sample = shap.sample(X, 100)
        explainer = shap.KernelExplainer(pipeline.predict_proba, background_sample)

        return pipeline, explainer, df

    except Exception as e:
        st.error(f"Unable to load prediction model: {str(e)}")
        return None, None, None

# Firebase Authentication helper functions
def signup(email, password):
    """
    Create new Firebase user via REST API
    Returns: (success, user_id_or_error_message)
    """
    api_key = os.environ.get("FIREBASE_WEB_API_KEY")

    if not api_key:
        return False, "Firebase configuration missing"

    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={api_key}"

    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return True, data['localId']
        else:
            error_data = response.json()
            error_message = error_data.get('error', {}).get('message', 'UNKNOWN_ERROR')

            # Parse specific error codes
            if 'EMAIL_EXISTS' in error_message:
                return False, "EMAIL_EXISTS"
            elif 'INVALID_EMAIL' in error_message:
                return False, "INVALID_EMAIL"
            elif 'WEAK_PASSWORD' in error_message:
                return False, "WEAK_PASSWORD"
            else:
                return False, "UNKNOWN_ERROR"

    except Exception as e:
        return False, "Network error"

def login(email, password):
    """
    Authenticate existing user via REST API
    Returns: (success, user_id_or_error_message)
    """
    api_key = os.environ.get("FIREBASE_WEB_API_KEY")

    if not api_key:
        return False, "Firebase configuration missing"

    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"

    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return True, data['localId']
        else:
            error_data = response.json()
            error_message = error_data.get('error', {}).get('message', 'UNKNOWN_ERROR')

            # Parse specific error codes
            if 'EMAIL_NOT_FOUND' in error_message or 'INVALID_PASSWORD' in error_message:
                return False, "INVALID_CREDENTIALS"
            elif 'USER_DISABLED' in error_message:
                return False, "USER_DISABLED"
            else:
                return False, "UNKNOWN_ERROR"

    except Exception as e:
        return False, "Network error"

def main():
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None

    # Check for invalid session state
    if st.session_state.logged_in and st.session_state.user_id is None:
        st.session_state.logged_in = False
        st.info("Session expired. Please login again.")

    # Check if logged in
    if not st.session_state.logged_in:
        # Show authentication UI
        show_auth_ui()
    else:
        # Show authenticated app
        show_authenticated_app()

def show_auth_ui():
    """Display login/signup forms"""
    # Sidebar navigation
    auth_mode = st.sidebar.radio("Choose Action", ["Login", "Sign Up"])

    # Center the form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.title("Welcome to CardioCheck ❤️")

        if auth_mode == "Login":
            st.subheader("Login to Your Account")

            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")

            if st.button("Login"):
                if email and password:
                    success, result = login(email, password)

                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_id = result
                        st.session_state.user_email = email
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        # Show error based on error code
                        if result == "INVALID_CREDENTIALS":
                            st.error("Invalid email or password")
                        elif result == "USER_DISABLED":
                            st.error("Account has been disabled. Contact support.")
                        elif result == "Network error":
                            st.error("Network error. Please check your connection and try again.")
                        elif result == "Firebase configuration missing":
                            st.error("Firebase configuration missing. Please contact administrator.")
                        else:
                            st.error("Login failed. Please try again.")
                else:
                    st.error("Please enter both email and password")

        else:  # Sign Up
            st.subheader("Create New Account")

            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")

            if st.button("Sign Up"):
                if email and password and confirm_password:
                    # Validate passwords match
                    if password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        success, result = signup(email, password)

                        if success:
                            st.session_state.logged_in = True
                            st.session_state.user_id = result
                            st.session_state.user_email = email
                            st.success("Account created successfully!")
                            st.rerun()
                        else:
                            # Show error based on error code
                            if result == "EMAIL_EXISTS":
                                st.error("Email already in use. Please login instead.")
                            elif result == "WEAK_PASSWORD":
                                st.error("Password too weak. Use at least 6 characters.")
                            elif result == "INVALID_EMAIL":
                                st.error("Invalid email format.")
                            elif result == "Network error":
                                st.error("Network error. Please check your connection and try again.")
                            elif result == "Firebase configuration missing":
                                st.error("Firebase configuration missing. Please contact administrator.")
                            else:
                                st.error("Signup failed. Please try again.")
                else:
                    st.error("Please fill all fields")

def show_authenticated_app():
    """Display the authenticated application with navigation"""
    # Sidebar
    st.sidebar.title("Welcome!")
    st.sidebar.write(f"Logged in as: {st.session_state.user_email}")

    # Navigation menu
    page = st.sidebar.radio("Menu", ["🏠 Home", "🩺 Predictor"])

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.user_email = None
        st.sidebar.success("Logged out successfully")
        st.rerun()

    # Load model
    model, explainer, df = train_model()

    if model is None:
        st.error("Unable to load prediction model. Please try again later.")
        return

    # Show selected page
    if page == "🏠 Home":
        show_home_page(df)
    else:
        show_predictor_page(model, explainer, df)

def show_home_page(df):
    """Display home dashboard with metrics and visualizations"""
    st.title("Welcome to CardioCheck! 🏠")
    st.write("Your personal cardiovascular health dashboard")

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Patients in Dataset", len(df))

    with col2:
        avg_chol = round(df['chol'].mean(), 1)
        st.metric("Average Cholesterol", f"{avg_chol} mg/dl")

    with col3:
        at_risk_pct = round((df['target'].sum() / len(df) * 100), 1)
        st.metric("At-Risk Patients (%)", f"{at_risk_pct}%")

    # Visualizations section
    st.subheader("📊 Dataset Insights")

    # Visualization 1: Age Distribution Histogram
    fig_age = go.Figure(data=[go.Histogram(x=df['age'], marker_color='#FF6B6B')])
    fig_age.update_layout(
        title="Age Distribution of Patients",
        xaxis_title="Age",
        yaxis_title="Count",
        showlegend=False
    )
    st.plotly_chart(fig_age, use_container_width=True)

    # Visualization 2: Feature Correlation Heatmap
    corr_matrix = df.corr()
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 8}
    ))
    fig_corr.update_layout(title="Feature Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

    # Visualization 3: Target Distribution Pie Chart
    target_counts = df['target'].value_counts()
    fig_pie = go.Figure(data=[go.Pie(
        labels=["No Disease (0)", "Heart Disease (1)"],
        values=[target_counts[0], target_counts[1]],
        marker_colors=["#90EE90", "#FF6B6B"],
        textinfo='label+percent'
    )])
    fig_pie.update_layout(title="Heart Disease Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

def show_predictor_page(model, explainer, df):
    """Display predictor page with input form and results"""
    st.title("🩺 Your Personal Risk Predictor")
    st.write("Enter your health information to assess cardiovascular risk")

    # Input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        # Column 1 inputs
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            sex_val = 1 if sex == "Male" else 0

            cp = st.selectbox("Chest Pain Type", [
                "Typical Angina (0)",
                "Atypical Angina (1)",
                "Non-anginal Pain (2)",
                "Asymptomatic (3)"
            ])
            cp_val = int(cp.split("(")[1].split(")")[0])

            trestbps = st.number_input("Resting Blood Pressure (mm Hg)",
                                       min_value=80, max_value=200, value=120)

        # Column 2 inputs
        with col2:
            chol = st.number_input("Serum Cholesterol (mg/dl)",
                                  min_value=100, max_value=600, value=200)

            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            fbs_val = 1 if fbs == "Yes" else 0

            restecg = st.selectbox("Resting ECG Results", [
                "Normal (0)",
                "ST-T Abnormality (1)",
                "LV Hypertrophy (2)"
            ])
            restecg_val = int(restecg.split("(")[1].split(")")[0])

            thalach = st.number_input("Maximum Heart Rate Achieved",
                                     min_value=60, max_value=220, value=150)

        # Column 3 inputs
        with col3:
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            exang_val = 1 if exang == "Yes" else 0

            oldpeak = st.number_input("ST Depression Induced by Exercise",
                                     min_value=0.0, max_value=10.0, value=1.0, step=0.1)

            slope = st.selectbox("Slope of Peak Exercise ST Segment", [
                "Upsloping (0)",
                "Flat (1)",
                "Downsloping (2)"
            ])
            slope_val = int(slope.split("(")[1].split(")")[0])

            ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])

            thal = st.selectbox("Thalassemia", [
                "Normal (1)",
                "Fixed Defect (2)",
                "Reversible Defect (3)"
            ])
            thal_val = int(thal.split("(")[1].split(")")[0])

        # Submit button
        submitted = st.form_submit_button("🔍 Predict Risk")

    # Process prediction when form is submitted
    if submitted:
        # Create input DataFrame
        input_data = {
            'age': age,
            'sex': sex_val,
            'cp': cp_val,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs_val,
            'restecg': restecg_val,
            'thalach': thalach,
            'exang': exang_val,
            'oldpeak': oldpeak,
            'slope': slope_val,
            'ca': ca,
            'thal': thal_val
        }
        input_df = pd.DataFrame([input_data])

        # Make prediction
        try:
            proba = model.predict_proba(input_df)
            risk_prob = proba[0][1]
            risk_score = risk_prob * 100

            # Generate SHAP values (may take time)
            with st.spinner("Analyzing risk factors..."):
                shap_values = explainer.shap_values(input_df)

            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Your Risk Score",
                "❗ Risk Factor Analysis",
                "📈 Comparison",
                "🗓️ Prediction History"
            ])

            # TAB 1: Risk Score Gauge
            with tab1:
                # Create gauge chart
                if risk_score < 33:
                    gauge_color = "#90EE90"
                elif risk_score < 66:
                    gauge_color = "#FFD700"
                else:
                    gauge_color = "#FF6B6B"

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_score,
                    title={'text': "Cardiovascular Risk Score (%)"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': gauge_color},
                        'steps': [
                            {'range': [0, 33], 'color': "#E8F5E9"},
                            {'range': [33, 66], 'color': "#FFF9C4"},
                            {'range': [66, 100], 'color': "#FFEBEE"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 66
                        }
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Risk interpretation
                if risk_score < 33:
                    st.success("✅ Low Risk: Your cardiovascular risk is low. Keep up the healthy lifestyle!")
                elif risk_score < 66:
                    st.warning("⚠️ Moderate Risk: Consider lifestyle changes and consult your doctor.")
                else:
                    st.error("🚨 High Risk: Please consult a healthcare professional immediately.")

            # TAB 2: SHAP Waterfall Plot
            with tab2:
                st.write("This chart shows how each of your health factors influenced your risk score.")
                st.write("Red bars increase risk, blue bars decrease risk. Longer bars = stronger influence.")

                try:
                    # Create SHAP waterfall plot
                    fig_shap, ax = plt.subplots()
                    shap.waterfall_plot(shap.Explanation(
                        values=shap_values[1][0],
                        base_values=explainer.expected_value[1],
                        data=input_df.iloc[0].values,
                        feature_names=input_df.columns.tolist()
                    ), show=False)
                    st.pyplot(fig_shap)
                except Exception as e:
                    st.warning("Feature analysis unavailable. Risk score is still accurate.")

            # TAB 3: Radar Chart Comparison
            with tab3:
                st.write("Compare your health metrics to average healthy and at-risk profiles.")

                # Prepare normalized data
                feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                               'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

                # Calculate min-max for normalization
                def normalize_values(values, df, features):
                    normalized = []
                    for i, feat in enumerate(features):
                        min_val = df[feat].min()
                        max_val = df[feat].max()
                        norm_val = (values[i] - min_val) / (max_val - min_val) if max_val > min_val else 0
                        normalized.append(norm_val)
                    return normalized

                # User's values
                user_values = [input_data[f] for f in feature_names]
                user_normalized = normalize_values(user_values, df, feature_names)

                # Healthy average (target=0)
                healthy_avg = [df[df['target']==0][f].mean() for f in feature_names]
                healthy_normalized = normalize_values(healthy_avg, df, feature_names)

                # At-risk average (target=1)
                risk_avg = [df[df['target']==1][f].mean() for f in feature_names]
                risk_normalized = normalize_values(risk_avg, df, feature_names)

                # Create radar chart
                fig_radar = go.Figure()

                fig_radar.add_trace(go.Scatterpolar(
                    r=user_normalized,
                    theta=feature_names,
                    fill='toself',
                    name='Your Profile',
                    line_color='blue'
                ))

                fig_radar.add_trace(go.Scatterpolar(
                    r=healthy_normalized,
                    theta=feature_names,
                    fill='toself',
                    name='Healthy Average',
                    line_color='green',
                    opacity=0.3
                ))

                fig_radar.add_trace(go.Scatterpolar(
                    r=risk_normalized,
                    theta=feature_names,
                    fill='toself',
                    name='At-Risk Average',
                    line_color='red',
                    opacity=0.3
                ))

                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Your Profile vs. Average Profiles",
                    showlegend=True
                )

                st.plotly_chart(fig_radar, use_container_width=True)

            # TAB 4: Prediction History
            with tab4:
                show_prediction_history()

            # Save prediction to Firestore
            save_prediction_to_firestore(input_data, risk_score)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

def show_prediction_history():
    """Display user's prediction history from Firestore"""
    try:
        db = firestore.client()
        predictions_ref = db.collection('users').document(st.session_state.user_id).collection('predictions')
        docs = predictions_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10).stream()

        predictions = []
        for doc in docs:
            predictions.append(doc.to_dict())

        if not predictions:
            st.info("No prediction history yet. Make your first prediction!")
        else:
            st.subheader("Your Recent Predictions")

            for pred in predictions:
                timestamp = pred.get('timestamp')
                risk_score = pred.get('risk_score', 0)

                # Format timestamp
                if timestamp:
                    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    timestamp_str = "Unknown date"

                with st.expander(f"Prediction on {timestamp_str}"):
                    st.metric("Risk Score", f"{risk_score:.1f}%")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Age:** {pred.get('age', 'N/A')}")
                        st.write(f"**Sex:** {'Male' if pred.get('sex', 0) == 1 else 'Female'}")
                    with col2:
                        st.write(f"**Cholesterol:** {pred.get('chol', 'N/A')} mg/dl")
                        st.write(f"**Blood Pressure:** {pred.get('trestbps', 'N/A')} mm Hg")
                    with col3:
                        st.write(f"**Max Heart Rate:** {pred.get('thalach', 'N/A')} bpm")

    except Exception as e:
        st.error(f"Unable to load prediction history: {str(e)}")

def save_prediction_to_firestore(input_data, risk_score):
    """Save prediction to user's Firestore collection"""
    try:
        db = firestore.client()

        prediction_data = {
            'timestamp': firestore.SERVER_TIMESTAMP,
            'risk_score': float(risk_score),
            'age': int(input_data['age']),
            'sex': int(input_data['sex']),
            'cp': int(input_data['cp']),
            'trestbps': int(input_data['trestbps']),
            'chol': int(input_data['chol']),
            'fbs': int(input_data['fbs']),
            'restecg': int(input_data['restecg']),
            'thalach': int(input_data['thalach']),
            'exang': int(input_data['exang']),
            'oldpeak': float(input_data['oldpeak']),
            'slope': int(input_data['slope']),
            'ca': int(input_data['ca']),
            'thal': int(input_data['thal'])
        }

        db.collection('users').document(st.session_state.user_id).collection('predictions').add(prediction_data)
        st.success("✅ Prediction saved to your history!")

    except Exception as e:
        st.warning("Prediction complete, but couldn't save to history. Results are still valid.")
        print(f"Firestore save error: {str(e)}")

if __name__ == "__main__":
    main()
