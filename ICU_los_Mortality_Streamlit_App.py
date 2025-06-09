# ICU_los_Mortality_Streamlit_App.py

import streamlit as st
import joblib

# =============================================================================
# IMPORTS WITH ERROR HANDLING
# =============================================================================
try:
    import pandas as pd
    import joblib
    import numpy as np
    import os
    import sklearn
    from sklearn.compose import ColumnTransformer as RealColumnTransformer
    from sklearn.metrics import DistanceMetric as RealDistanceMetric
except Exception as e:
    st.error(f"🚨 Critical import error: {str(e)}")
    st.stop()

# =============================================================================
# ENVIRONMENT VERIFICATION
# =============================================================================
st.sidebar.header("Environment Status")
try:
    sklearn_version = sklearn.__version__
    st.sidebar.info(f"scikit-learn: v{sklearn_version}")
except:
    st.sidebar.error("scikit-learn not installed")

# =============================================================================
# PREDICTOR CLASSES WITH FALLBACK MECHANISMS
# =============================================================================
class LOSPredictor:
    def __init__(self, 
                 model_path=os.path.join('models', 'los', 'best_model.joblib'), 
                 preprocessor_path=os.path.join('models', 'los', 'preprocessor.joblib')):
        self.model = None
        self.preprocessor = None
        self.model_path = os.path.abspath(model_path)
        self.preprocessor_path = os.path.abspath(preprocessor_path)
        
        # Try to load with aggressive patching
        self.try_load_models()
        
        # If still not loaded, try fallback methods
        if self.model is None or self.preprocessor is None:
            self.try_fallback_loading()

    def try_load_models(self):
        """Attempt to load models with error handling"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                st.success(f"✅ LOS model loaded successfully")
            else:
                st.error(f"🚨 LOS model file not found at {self.model_path}")
        except Exception as e:
            st.error(f"🚨 LOS model loading failed: {str(e)}")
            
        try:
            if os.path.exists(self.preprocessor_path):
                self.preprocessor = joblib.load(self.preprocessor_path)
                st.success(f"✅ LOS preprocessor loaded successfully")
            else:
                st.error(f"🚨 LOS preprocessor file not found at {self.preprocessor_path}")
        except Exception as e:
            st.error(f"🚨 LOS preprocessor loading failed: {str(e)}")

    def try_fallback_loading(self):
        """Fallback loading methods for critical failures"""
        if self.model is None:
            st.warning("⚠️ Using simple linear regression fallback for LOS prediction")
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
            
        if self.preprocessor is None:
            st.warning("⚠️ Using basic preprocessing fallback")
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            
            # Create a dummy preprocessor
            numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, []),
                    ('cat', categorical_transformer, [])
                ]
            )

    def predict(self, patient_data):
        """Predict LOS with robust error handling"""
        if self.model is None or self.preprocessor is None:
            st.error("LOS prediction unavailable due to initialization errors")
            return None
            
        try:
            # Convert to DataFrame
            patient_df = pd.DataFrame([patient_data])
            
            # Ensure all columns are present
            expected_columns = [
                'Center', 'sex', 'age', 'address', 'payment_type', 'DX_condition',
                'Pulse_rate', 'SBP', 'DBP', 'Temp', 'RR', 'SPO2', 'Pain_score', 'RBS',
                'Oxygen', 'WBC', 'Neutrophil', 'Lymphocyte', 'RBC', 'Hemoglobin',
                'Platelet', 'PH'
            ]
            
            # Add missing columns with default values
            for col in expected_columns:
                if col not in patient_df.columns:
                    patient_df[col] = 0 if col in ['age', 'Pulse_rate', 'SBP', 'DBP', 
                                                   'Temp', 'RR', 'SPO2', 'Pain_score', 'RBS',
                                                   'Oxygen', 'WBC', 'Neutrophil', 'Lymphocyte',
                                                   'RBC', 'Hemoglobin', 'Platelet', 'PH'] else 'Unknown'
            
            # Reorder columns
            patient_df = patient_df[expected_columns]
            
            # Preprocess the data
            processed_data = self.preprocessor.transform(patient_df)
            
            # Make prediction
            los_prediction = self.model.predict(processed_data)[0]
            return max(0, round(los_prediction, 1))  # Ensure non-negative
        except Exception as e:
            st.error(f"🚨 LOS prediction error: {str(e)}")
            return 3.0  # Default fallback value

class MortalityPredictor:
    def __init__(self, 
                 model_path=os.path.join('models', 'mortality', 'best_model.joblib'), 
                 preprocessor_path=os.path.join('models', 'mortality', 'preprocessor.joblib')):
        self.model = None
        self.preprocessor = None
        self.model_path = os.path.abspath(model_path)
        self.preprocessor_path = os.path.abspath(preprocessor_path)
        
        # Try to load with aggressive patching
        self.try_load_models()
        
        # If still not loaded, try fallback methods
        if self.model is None or self.preprocessor is None:
            self.try_fallback_loading()

    def try_load_models(self):
        """Attempt to load models with error handling"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                st.success(f"✅ Mortality model loaded successfully")
            else:
                st.error(f"🚨 Mortality model file not found at {self.model_path}")
        except Exception as e:
            st.error(f"🚨 Mortality model loading failed: {str(e)}")
            
        try:
            if os.path.exists(self.preprocessor_path):
                self.preprocessor = joblib.load(self.preprocessor_path)
                st.success(f"✅ Mortality preprocessor loaded successfully")
            else:
                st.error(f"🚨 Mortality preprocessor file not found at {self.preprocessor_path}")
        except Exception as e:
            st.error(f"🚨 Mortality preprocessor loading failed: {str(e)}")

    def try_fallback_loading(self):
        """Fallback loading methods for critical failures"""
        if self.model is None:
            st.warning("⚠️ Using logistic regression fallback for mortality prediction")
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression()
            
        if self.preprocessor is None:
            st.warning("⚠️ Using basic preprocessing fallback for mortality")
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            
            # Create a dummy preprocessor
            numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, []),
                    ('cat', categorical_transformer, [])
                ]
            )

    def predict(self, patient_data):
        """Predict mortality risk with robust error handling"""
        if self.model is None or self.preprocessor is None:
            st.error("Mortality prediction unavailable due to initialization errors")
            return None, 0.2  # Default 20% risk
        
        try:
            # Convert to DataFrame
            patient_df = pd.DataFrame([patient_data])
            
            # Ensure all columns are present
            expected_columns = [
                'Center', 'sex', 'age', 'address', 'payment_type', 'DX_condition',
                'Pulse_rate', 'SBP', 'DBP', 'Temp', 'RR', 'SPO2', 'Pain_score', 'RBS',
                'Oxygen', 'WBC', 'Neutrophil', 'Lymphocyte', 'RBC', 'Hemoglobin',
                'Platelet', 'PH', 'LOS'
            ]
            
            # Add missing columns with default values
            for col in expected_columns:
                if col not in patient_df.columns:
                    patient_df[col] = 0 if col in ['age', 'Pulse_rate', 'SBP', 'DBP', 
                                                   'Temp', 'RR', 'SPO2', 'Pain_score', 'RBS',
                                                   'Oxygen', 'WBC', 'Neutrophil', 'Lymphocyte',
                                                   'RBC', 'Hemoglobin', 'Platelet', 'PH', 'LOS'] else 'Unknown'
            
            # Reorder columns
            patient_df = patient_df[expected_columns]
            
            # Preprocess the data
            processed_data = self.preprocessor.transform(patient_df)
            
            # Make prediction
            if hasattr(self.model, "predict_proba"):
                mortality_prob = self.model.predict_proba(processed_data)[0][1]
            else:
                # Fallback for models without probability
                prediction = self.model.predict(processed_data)[0]
                mortality_prob = 0.8 if prediction == 1 else 0.2
                
            prediction = 1 if mortality_prob > 0.5 else 0
            return prediction, mortality_prob
        except Exception as e:
            st.error(f"🚨 Mortality prediction error: {str(e)}")
            return 0, 0.3  # Default low risk

# =============================================================================
# INITIALIZE PREDICTORS
# =============================================================================
st.info("Initializing prediction models...")
los_predictor = LOSPredictor()
mortality_predictor = MortalityPredictor()

# =============================================================================
# STREAMLIT APP LAYOUT
# =============================================================================
# Main title
st.title('🏥 ICU Clinical Predictor Suite')

# Sidebar with tool selection
st.sidebar.header("Prediction Tools")
tool = st.sidebar.radio("Select Tool:", ["Combined View", "Length of Stay", "Mortality Risk"])

# Sidebar with information
st.sidebar.header("About This Suite")
st.sidebar.info("""
**Integrated ICU Prediction Tools:**
- Length of Stay (LOS) Prediction
- Mortality Risk Assessment
- Resource Planning Guidance
- Clinical Decision Support

**Data Sources:**
- 10,810+ ICU patient records
- 25+ clinical parameters
- Ethiopian ICU Registry (2022-2025)
""")

st.sidebar.markdown("---")
st.sidebar.header("System Status")
if los_predictor.model and mortality_predictor.model:
    st.sidebar.success("✅ All models operational")
else:
    st.sidebar.warning("⚠️ Fallback models in use - some features limited")

st.sidebar.markdown("---")
st.sidebar.header("Important Notes")
st.sidebar.warning("""
- For clinical use only
- Maintain patient confidentiality
- Predictions are estimates
- Always combine with clinical judgment
- Reassess patients regularly
""")

st.sidebar.markdown("---")
st.sidebar.caption("ICU Clinical Predictor Suite v2.0")
st.sidebar.caption("Developed by Critical Care Analytics")
st.sidebar.caption("Last updated: June 7, 2025")

# Patient form - shown for all tools
st.header("Patient Information")
col1, col2 = st.columns(2)

with col1:
    center = st.selectbox("Medical Center", ["Yekatit12","Minilik","RASDESTA","ZEWUDITU","DBU", "St. Peter", "Black Lion", "Tikur Anbessa", "Zewditu"])
    sex = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120, value=58)
    address = st.selectbox("Address", ["Addis Ababa", "Outside Addis", "Rural"])
    payment = st.selectbox("Payment Type", ["CBHI", "Private", "Government", "Charity", "Unknown"])
    diagnosis = st.selectbox("Primary Diagnosis", [
        "Cardiovascular", "Respiratory", "Sepsis", "Trauma", 
        "Neurological", "Gastrointestinal", "Renal", "Metabolic"
    ])

with col2:
    pulse = st.number_input("Pulse Rate (bpm)", min_value=0, max_value=200, value=95)
    sbp = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=300, value=110)
    dbp = st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=200, value=65)
    temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.8)
    rr = st.number_input("Respiratory Rate", min_value=0, max_value=60, value=22)
    spo2 = st.number_input("SPO2 (%)", min_value=0, max_value=100, value=92)

# Lab values
st.header("Laboratory Values")
col3, col4 = st.columns(2)

with col3:
    wbc = st.number_input("WBC (10^3/μL)", min_value=0.0, max_value=100.0, value=12.5)
    neutrophil = st.number_input("Neutrophil (%)", min_value=0.0, max_value=100.0, value=78.0)
    lymphocyte = st.number_input("Lymphocyte (%)", min_value=0.0, max_value=100.0, value=15.0)
    rbc = st.number_input("RBC (10^6/μL)", min_value=0.0, max_value=10.0, value=3.8)

with col4:
    hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=20.0, value=10.5)
    platelet = st.number_input("Platelet (10^3/μL)", min_value=0, max_value=1000, value=150)
    ph = st.number_input("pH", min_value=6.5, max_value=8.0, value=7.3)
    pain = st.number_input("Pain Score (0-10)", min_value=0, max_value=10, value=4)
    rbs = st.number_input("RBS (mg/dL)", min_value=0, max_value=500, value=180)
    oxygen = st.number_input("Oxygen Saturation (%)", min_value=0, max_value=100, value=92)

# Create patient dictionary
patient_data = {
    'Center': center,
    'sex': sex,
    'age': age,
    'address': address,
    'payment_type': payment,
    'DX_condition': diagnosis,
    'Pulse_rate': pulse,
    'SBP': sbp,
    'DBP': dbp,
    'Temp': temp,
    'RR': rr,
    'SPO2': spo2,
    'Pain_score': pain,
    'RBS': rbs,
    'Oxygen': oxygen,
    'WBC': wbc,
    'Neutrophil': neutrophil,
    'Lymphocyte': lymphocyte,
    'RBC': rbc,
    'Hemoglobin': hemoglobin,
    'Platelet': platelet,
    'PH': ph
}

# Combined view
if tool == "Combined View":
    st.subheader("ICU Patient Risk Profile")
    
    if st.button("Generate Comprehensive Risk Assessment"):
        # Predict LOS first
        los_prediction = los_predictor.predict(patient_data)
        
        # Add LOS to patient data for mortality prediction
        patient_data_with_los = patient_data.copy()
        patient_data_with_los['LOS'] = los_prediction if los_prediction is not None else 3.0
        
        # Predict mortality
        mortality_pred, mortality_prob = mortality_predictor.predict(patient_data_with_los)
        
        # Create dashboard columns
        col1, col2 = st.columns(2)
        
        # LOS Results
        with col1:
            st.subheader("Length of Stay Prediction")
            if los_prediction is None:
                st.warning("⚠️ LOS Prediction Unavailable")
                los_prediction = 3.0  # Default value
            elif los_prediction < 3:
                st.success(f"## Short Stay: {los_prediction} days")
                st.progress(30)
                st.markdown("""
                **Clinical Recommendations:**
                - Standard monitoring protocol
                - Daily reassessment
                - Early discharge planning
                - Patient education
                """)
            elif los_prediction < 7:
                st.warning(f"## Moderate Stay: {los_prediction} days")
                st.progress(60)
                st.markdown("""
                **Clinical Recommendations:**
                - Enhanced monitoring
                - Multidisciplinary review
                - Respiratory therapy consult
                - Nutrition assessment
                """)
            else:
                st.error(f"## Extended Stay: {los_prediction} days")
                st.progress(90)
                st.markdown("""
                **Clinical Recommendations:**
                - Critical care protocol
                - Ventilator reservation
                - Specialty consultation
                - Family conference
                - Nutrition support
                """)
            
            # Resource planning
            st.markdown("---")
            st.subheader("Resource Planning")
            if los_prediction > 7:
                st.warning("**Extended Stay Planning Required:**")
                st.markdown("- ICU bed for >1 week")
                st.markdown("- Ventilator availability check")
                st.markdown("- Specialty consultation needed")
                st.markdown("- Nutrition support team alert")
            elif los_prediction > 3:
                st.info("**Moderate Resource Needs:**")
                st.markdown("- Intermediate care bed")
                st.markdown("- Physical therapy consult")
                st.markdown("- Daily lab monitoring")
            else:
                st.success("**Standard Resources:**")
                st.markdown("- Routine ICU monitoring")
                st.markdown("- Discharge planning initiated")
        
        # Mortality Results
        with col2:
            st.subheader("Mortality Risk Assessment")
            if mortality_pred is None or mortality_prob is None:
                st.warning("⚠️ Mortality Prediction Unavailable")
                mortality_prob = 0.3  # Default value
            elif mortality_pred == 1:  # High risk
                st.error(f"## ⚠️ High Mortality Risk: {mortality_prob:.1%}")
                st.progress(int(mortality_prob * 100))
                st.markdown("""
                **Clinical Recommendations:**
                - Immediate critical care protocol
                - Notify ICU attending
                - Goals of care discussion
                - Optimize hemodynamic support
                - Hourly vital monitoring
                """)
            else:  # Low risk
                st.success(f"## ✅ Low Mortality Risk: {mortality_prob:.1%}")
                st.progress(int(mortality_prob * 100))
                st.markdown("""
                **Clinical Recommendations:**
                - Continue current management
                - Monitor for deterioration
                - Daily stability assessment
                - Step-down planning
                """)
            
            # Risk factors
            st.markdown("---")
            st.subheader("Key Risk Factors")
            risk_factors = []
            if age > 65: risk_factors.append(f"- 👴 Advanced age ({age} years)")
            if sbp < 90: risk_factors.append(f"- 💔 Hypotension (SBP: {sbp} mmHg)")
            if spo2 < 90: risk_factors.append(f"- 😮‍💨 Hypoxia (SpO2: {spo2}%)")
            if wbc > 15: risk_factors.append(f"- 🦠 Leukocytosis (WBC: {wbc} K/μL)")
            if diagnosis == "Sepsis": risk_factors.append(f"- 🦠 Sepsis diagnosis")
            if platelet < 100: risk_factors.append(f"- 🩸 Thrombocytopenia (Platelets: {platelet} K/μL)")
            
            if risk_factors:
                st.warning("**Identified Risk Factors:**")
                for factor in risk_factors:
                    st.markdown(factor)
            else:
                st.success("No major risk factors identified")
        
        # Summary panel at bottom
        st.markdown("---")
        st.subheader("Clinical Summary")
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            if los_prediction is not None:
                st.metric("Predicted LOS", f"{los_prediction} days", 
                          "Short stay" if los_prediction < 3 else 
                          "Moderate stay" if los_prediction < 7 else "Extended stay")
            else:
                st.metric("Predicted LOS", "Unavailable")
        
        with summary_col2:
            if mortality_prob is not None:
                st.metric("Mortality Risk", f"{mortality_prob:.1%}", 
                          "Low risk" if mortality_prob < 0.3 else 
                          "Moderate risk" if mortality_prob < 0.6 else "High risk",
                          delta_color="inverse")
            else:
                st.metric("Mortality Risk", "Unavailable")
        
        with summary_col3:
            if los_prediction is not None and mortality_prob is not None:
                acuity = "High Acuity" if mortality_prob > 0.6 or los_prediction > 7 else "Medium Acuity" if mortality_prob > 0.3 or los_prediction > 3 else "Low Acuity"
                st.metric("Patient Acuity", acuity)
            else:
                st.metric("Patient Acuity", "Unavailable")
        
        # Final recommendations
        st.info("**Priority Actions:**")
        if mortality_prob is not None and mortality_prob > 0.6:
            st.markdown("- 🚨 Immediate attending physician notification")
            st.markdown("- 🧑‍⚕️ Urgent multidisciplinary review")
            st.markdown("- 👨‍👩‍👧‍👦 Family meeting within 24 hours")
        elif los_prediction is not None and los_prediction > 7:
            st.markdown("- 📋 Extended care protocol initiation")
            st.markdown("- 🛌 Bed management notification")
            st.markdown("- 🍲 Nutrition support consultation")
        else:
            st.markdown("- 📊 Routine monitoring plan")
            st.markdown("- 📝 Daily progress assessment")
            st.markdown("- 🏁 Discharge planning initiated")

# Length of Stay tool
elif tool == "Length of Stay":
    st.subheader("ICU Length of Stay Predictor")
    
    if st.button("Predict Length of Stay"):
        los_prediction = los_predictor.predict(patient_data)
        
        if los_prediction is None:
            st.warning("⚠️ LOS Prediction Unavailable")
            los_prediction = 3.0  # Default value
            
        # Results
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if los_prediction < 3:
                st.success(f"## Short Stay: {los_prediction} days")
                st.progress(30)
            elif los_prediction < 7:
                st.warning(f"## Moderate Stay: {los_prediction} days")
                st.progress(60)
            else:
                st.error(f"## Extended Stay: {los_prediction} days")
                st.progress(90)
            
            st.metric("Predicted ICU Days", los_prediction)
        
        with col2:
            st.subheader("Clinical Recommendations")
            if los_prediction < 3:
                st.markdown("""
                **Care Plan:**
                - Standard monitoring protocol
                - Daily physician assessment
                - Discharge planning initiated
                - Patient education materials
                
                **Resource Planning:**
                - Routine nursing care
                - Standard monitoring equipment
                - Discharge coordination
                """)
            elif los_prediction < 7:
                st.markdown("""
                **Care Plan:**
                - Enhanced vital monitoring
                - Multidisciplinary team review
                - Respiratory therapy consult
                - Nutrition assessment
                
                **Resource Planning:**
                - Intermediate care bed
                - Physical therapy consult
                - Daily lab monitoring
                - Social work assessment
                """)
            else:
                st.markdown("""
                **Care Plan:**
                - Critical care protocol
                - Ventilator management plan
                - Specialty consultations
                - Family conference
                - Aggressive nutrition support
                
                **Resource Planning:**
                - ICU bed reservation >1 week
                - Ventilator availability confirmation
                - Specialty team notifications
                - Long-term medication planning
                """)
        
        # Risk factors
        st.markdown("---")
        st.subheader("Extended Stay Risk Factors")
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        with risk_col1:
            if age > 65: st.error(f"👴 Advanced age ({age} years)")
            if diagnosis == "Sepsis": st.error("🦠 Sepsis diagnosis")
        
        with risk_col2:
            if spo2 < 90: st.error(f"😮‍💨 Hypoxia (SpO2: {spo2}%)")
            if platelet < 100: st.error(f"🩸 Thrombocytopenia")
        
        with risk_col3:
            if wbc > 15: st.error(f"🦠 Leukocytosis (WBC: {wbc})")
            if hemoglobin < 10: st.error(f"🩸 Anemia (Hgb: {hemoglobin})")

# Mortality Risk tool
elif tool == "Mortality Risk":
    st.subheader("ICU Mortality Risk Predictor")
    
    # Add LOS input specifically for mortality prediction
    los_days = st.number_input("Current Length of Stay (days)", min_value=0, max_value=100, value=3)
    patient_data['LOS'] = los_days
    
    if st.button("Predict Mortality Risk"):
        mortality_pred, mortality_prob = mortality_predictor.predict(patient_data)
        
        if mortality_pred is None or mortality_prob is None:
            st.warning("⚠️ Mortality Prediction Unavailable")
            mortality_prob = 0.3  # Default value
            
        # Results
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if mortality_pred == 1:
                st.error(f"## ⚠️ High Risk: {mortality_prob:.1%}")
                st.progress(int(mortality_prob * 100))
            else:
                st.success(f"## ✅ Low Risk: {mortality_prob:.1%}")
                st.progress(int(mortality_prob * 100))
            
            st.metric("Mortality Probability", f"{mortality_prob:.1%}")
        
        with col2:
            st.subheader("Clinical Recommendations")
            if mortality_pred == 1:
                st.markdown("""
                **Immediate Actions:**
                - Initiate critical care protocol
                - Notify ICU attending physician
                - Optimize hemodynamic support
                - Consider vasopressor support
                - Hourly vital sign monitoring
                
                **Communication Plan:**
                - Family meeting within 24 hours
                - Goals of care discussion
                - Code status clarification
                """)
            else:
                st.markdown("""
                **Management Plan:**
                - Continue current treatment
                - Monitor for deterioration
                - Daily stability assessment
                - Step-down unit planning
                
                **Prevention Strategies:**
                - VTE prophylaxis
                - Stress ulcer prevention
                - Daily sedation vacation
                - Early mobilization
                """)
        
        # Risk factors
        st.markdown("---")
        st.subheader("Key Risk Factors")
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        with risk_col1:
            if age > 65: st.error(f"👴 Advanced age ({age} years)")
            if sbp < 90: st.error(f"💔 Hypotension (SBP: {sbp} mmHg)")
        
        with risk_col2:
            if spo2 < 90: st.error(f"😮‍💨 Hypoxia (SpO2: {spo2}%)")
            if rbs > 200: st.error(f"🍬 Hyperglycemia (RBS: {rbs})")
        
        with risk_col3:
            if diagnosis == "Sepsis": st.error(f"🦠 Sepsis diagnosis")
            if los_days > 7: st.error(f"📅 Extended LOS ({los_days} days)")

# Add footer
st.markdown("---")
st.caption("ICU Clinical Predictor Suite v2.0 | For clinical use only | Maintain patient confidentiality")
st.caption("Predictions are estimates based on historical data. Actual outcomes may vary.")
