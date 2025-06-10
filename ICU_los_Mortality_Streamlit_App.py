# # ICU_los_Mortality_Streamlit_App.py

# import streamlit as st
# import joblib

# # =============================================================================
# # IMPORTS WITH ERROR HANDLING
# # =============================================================================
# try:
#     import pandas as pd
#     import joblib
#     import numpy as np
#     import os
#     import sklearn
#     from sklearn.compose import ColumnTransformer as RealColumnTransformer
#     from sklearn.metrics import DistanceMetric as RealDistanceMetric
# except Exception as e:
#     st.error(f"🚨 Critical import error: {str(e)}")
#     st.stop()

# # =============================================================================
# # ENVIRONMENT VERIFICATION
# # =============================================================================
# st.sidebar.header("Environment Status")
# try:
#     sklearn_version = sklearn.__version__
#     st.sidebar.info(f"scikit-learn: v{sklearn_version}")
# except:
#     st.sidebar.error("scikit-learn not installed")

# # =============================================================================
# # PREDICTOR CLASSES WITH FALLBACK MECHANISMS
# # =============================================================================
# class LOSPredictor:
#     def __init__(self, 
#                  model_path=os.path.join('models', 'los', 'best_model.joblib'), 
#                  preprocessor_path=os.path.join('models', 'los', 'preprocessor.joblib')):
#         self.model = None
#         self.preprocessor = None
#         self.model_path = os.path.abspath(model_path)
#         self.preprocessor_path = os.path.abspath(preprocessor_path)
        
#         # Try to load with aggressive patching
#         self.try_load_models()
        
#         # If still not loaded, try fallback methods
#         if self.model is None or self.preprocessor is None:
#             self.try_fallback_loading()

#     def try_load_models(self):
#         """Attempt to load models with error handling"""
#         try:
#             if os.path.exists(self.model_path):
#                 self.model = joblib.load(self.model_path)
#                 st.success(f"✅ LOS model loaded successfully")
#             else:
#                 st.error(f"🚨 LOS model file not found at {self.model_path}")
#         except Exception as e:
#             st.error(f"🚨 LOS model loading failed: {str(e)}")
            
#         try:
#             if os.path.exists(self.preprocessor_path):
#                 self.preprocessor = joblib.load(self.preprocessor_path)
#                 st.success(f"✅ LOS preprocessor loaded successfully")
#             else:
#                 st.error(f"🚨 LOS preprocessor file not found at {self.preprocessor_path}")
#         except Exception as e:
#             st.error(f"🚨 LOS preprocessor loading failed: {str(e)}")

#     def try_fallback_loading(self):
#         """Fallback loading methods for critical failures"""
#         if self.model is None:
#             st.warning("⚠️ Using simple linear regression fallback for LOS prediction")
#             from sklearn.linear_model import LinearRegression
#             self.model = LinearRegression()
            
#         if self.preprocessor is None:
#             st.warning("⚠️ Using basic preprocessing fallback")
#             from sklearn.preprocessing import StandardScaler, OneHotEncoder
#             from sklearn.compose import ColumnTransformer
#             from sklearn.pipeline import Pipeline
            
#             # Create a dummy preprocessor
#             numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
#             categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
            
#             self.preprocessor = ColumnTransformer(
#                 transformers=[
#                     ('num', numeric_transformer, []),
#                     ('cat', categorical_transformer, [])
#                 ]
#             )

#     def predict(self, patient_data):
#         """Predict LOS with robust error handling"""
#         if self.model is None or self.preprocessor is None:
#             st.error("LOS prediction unavailable due to initialization errors")
#             return None
            
#         try:
#             # Convert to DataFrame
#             patient_df = pd.DataFrame([patient_data])
            
#             # Ensure all columns are present
#             expected_columns = [
#                 'Center', 'sex', 'age', 'address', 'payment_type', 'DX_condition',
#                 'Pulse_rate', 'SBP', 'DBP', 'Temp', 'RR', 'SPO2', 'Pain_score', 'RBS',
#                 'Oxygen', 'WBC', 'Neutrophil', 'Lymphocyte', 'RBC', 'Hemoglobin',
#                 'Platelet', 'PH'
#             ]
            
#             # Add missing columns with default values
#             for col in expected_columns:
#                 if col not in patient_df.columns:
#                     patient_df[col] = 0 if col in ['age', 'Pulse_rate', 'SBP', 'DBP', 
#                                                    'Temp', 'RR', 'SPO2', 'Pain_score', 'RBS',
#                                                    'Oxygen', 'WBC', 'Neutrophil', 'Lymphocyte',
#                                                    'RBC', 'Hemoglobin', 'Platelet', 'PH'] else 'Unknown'
            
#             # Reorder columns
#             patient_df = patient_df[expected_columns]
            
#             # Preprocess the data
#             processed_data = self.preprocessor.transform(patient_df)
            
#             # Make prediction
#             los_prediction = self.model.predict(processed_data)[0]
#             return max(0, round(los_prediction, 1))  # Ensure non-negative
#         except Exception as e:
#             st.error(f"🚨 LOS prediction error: {str(e)}")
#             return 3.0  # Default fallback value

# class MortalityPredictor:
#     def __init__(self, 
#                  model_path=os.path.join('models', 'mortality', 'best_model.joblib'), 
#                  preprocessor_path=os.path.join('models', 'mortality', 'preprocessor.joblib')):
#         self.model = None
#         self.preprocessor = None
#         self.model_path = os.path.abspath(model_path)
#         self.preprocessor_path = os.path.abspath(preprocessor_path)
        
#         # Try to load with aggressive patching
#         self.try_load_models()
        
#         # If still not loaded, try fallback methods
#         if self.model is None or self.preprocessor is None:
#             self.try_fallback_loading()

#     def try_load_models(self):
#         """Attempt to load models with error handling"""
#         try:
#             if os.path.exists(self.model_path):
#                 self.model = joblib.load(self.model_path)
#                 st.success(f"✅ Mortality model loaded successfully")
#             else:
#                 st.error(f"🚨 Mortality model file not found at {self.model_path}")
#         except Exception as e:
#             st.error(f"🚨 Mortality model loading failed: {str(e)}")
            
#         try:
#             if os.path.exists(self.preprocessor_path):
#                 self.preprocessor = joblib.load(self.preprocessor_path)
#                 st.success(f"✅ Mortality preprocessor loaded successfully")
#             else:
#                 st.error(f"🚨 Mortality preprocessor file not found at {self.preprocessor_path}")
#         except Exception as e:
#             st.error(f"🚨 Mortality preprocessor loading failed: {str(e)}")

#     def try_fallback_loading(self):
#         """Fallback loading methods for critical failures"""
#         if self.model is None:
#             st.warning("⚠️ Using logistic regression fallback for mortality prediction")
#             from sklearn.linear_model import LogisticRegression
#             self.model = LogisticRegression()
            
#         if self.preprocessor is None:
#             st.warning("⚠️ Using basic preprocessing fallback for mortality")
#             from sklearn.preprocessing import StandardScaler, OneHotEncoder
#             from sklearn.compose import ColumnTransformer
#             from sklearn.pipeline import Pipeline
            
#             # Create a dummy preprocessor
#             numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
#             categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
            
#             self.preprocessor = ColumnTransformer(
#                 transformers=[
#                     ('num', numeric_transformer, []),
#                     ('cat', categorical_transformer, [])
#                 ]
#             )

#     def predict(self, patient_data):
#         """Predict mortality risk with robust error handling"""
#         if self.model is None or self.preprocessor is None:
#             st.error("Mortality prediction unavailable due to initialization errors")
#             return None, 0.2  # Default 20% risk
        
#         try:
#             # Convert to DataFrame
#             patient_df = pd.DataFrame([patient_data])
            
#             # Ensure all columns are present
#             expected_columns = [
#                 'Center', 'sex', 'age', 'address', 'payment_type', 'DX_condition',
#                 'Pulse_rate', 'SBP', 'DBP', 'Temp', 'RR', 'SPO2', 'Pain_score', 'RBS',
#                 'Oxygen', 'WBC', 'Neutrophil', 'Lymphocyte', 'RBC', 'Hemoglobin',
#                 'Platelet', 'PH', 'LOS'
#             ]
            
#             # Add missing columns with default values
#             for col in expected_columns:
#                 if col not in patient_df.columns:
#                     patient_df[col] = 0 if col in ['age', 'Pulse_rate', 'SBP', 'DBP', 
#                                                    'Temp', 'RR', 'SPO2', 'Pain_score', 'RBS',
#                                                    'Oxygen', 'WBC', 'Neutrophil', 'Lymphocyte',
#                                                    'RBC', 'Hemoglobin', 'Platelet', 'PH', 'LOS'] else 'Unknown'
            
#             # Reorder columns
#             patient_df = patient_df[expected_columns]
            
#             # Preprocess the data
#             processed_data = self.preprocessor.transform(patient_df)
            
#             # Make prediction
#             if hasattr(self.model, "predict_proba"):
#                 mortality_prob = self.model.predict_proba(processed_data)[0][1]
#             else:
#                 # Fallback for models without probability
#                 prediction = self.model.predict(processed_data)[0]
#                 mortality_prob = 0.8 if prediction == 1 else 0.2
                
#             prediction = 1 if mortality_prob > 0.5 else 0
#             return prediction, mortality_prob
#         except Exception as e:
#             st.error(f"🚨 Mortality prediction error: {str(e)}")
#             return 0, 0.3  # Default low risk

# # =============================================================================
# # INITIALIZE PREDICTORS
# # =============================================================================
# st.info("Initializing prediction models...")
# los_predictor = LOSPredictor()
# mortality_predictor = MortalityPredictor()

# # =============================================================================
# # STREAMLIT APP LAYOUT
# # =============================================================================
# # Main title
# st.title('🏥 ICU Clinical Predictor Suite')

# # Sidebar with tool selection
# st.sidebar.header("Prediction Tools")
# tool = st.sidebar.radio("Select Tool:", ["Combined View", "Length of Stay", "Mortality Risk"])

# # Sidebar with information
# st.sidebar.header("About This Suite")
# st.sidebar.info("""
# **Integrated ICU Prediction Tools:**
# - Length of Stay (LOS) Prediction
# - Mortality Risk Assessment
# - Resource Planning Guidance
# - Clinical Decision Support

# **Data Sources:**
# - 10,810+ ICU patient records
# - 25+ clinical parameters
# - Ethiopian ICU Registry (2022-2025)
# """)

# st.sidebar.markdown("---")
# st.sidebar.header("System Status")
# if los_predictor.model and mortality_predictor.model:
#     st.sidebar.success("✅ All models operational")
# else:
#     st.sidebar.warning("⚠️ Fallback models in use - some features limited")

# st.sidebar.markdown("---")
# st.sidebar.header("Important Notes")
# st.sidebar.warning("""
# - For clinical use only
# - Maintain patient confidentiality
# - Predictions are estimates
# - Always combine with clinical judgment
# - Reassess patients regularly
# """)

# st.sidebar.markdown("---")
# st.sidebar.caption("ICU Clinical Predictor Suite v2.0")
# st.sidebar.caption("Developed by Critical Care Analytics")
# st.sidebar.caption("Last updated: June 7, 2025")

# # Patient form - shown for all tools
# st.header("Patient Information")
# col1, col2 = st.columns(2)

# with col1:
#     center = st.selectbox("Medical Center", ["Yekatit12","Minilik","RASDESTA","ZEWUDITU","DBU", "St. Peter", "Black Lion", "Tikur Anbessa", "Zewditu"])
#     sex = st.selectbox("Gender", ["Male", "Female"])
#     age = st.number_input("Age", min_value=0, max_value=120, value=58)
#     address = st.selectbox("Address", ["Addis Ababa", "Outside Addis", "Rural"])
#     payment = st.selectbox("Payment Type", ["CBHI", "Private", "Government", "Charity", "Unknown"])
#     diagnosis = st.selectbox("Primary Diagnosis", [
#         "Cardiovascular", "Respiratory", "Sepsis", "Trauma", 
#         "Neurological", "Gastrointestinal", "Renal", "Metabolic"
#     ])

# with col2:
#     pulse = st.number_input("Pulse Rate (bpm)", min_value=0, max_value=200, value=95)
#     sbp = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=300, value=110)
#     dbp = st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=200, value=65)
#     temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.8)
#     rr = st.number_input("Respiratory Rate", min_value=0, max_value=60, value=22)
#     spo2 = st.number_input("SPO2 (%)", min_value=0, max_value=100, value=92)

# # Lab values
# st.header("Laboratory Values")
# col3, col4 = st.columns(2)

# with col3:
#     wbc = st.number_input("WBC (10^3/μL)", min_value=0.0, max_value=100.0, value=12.5)
#     neutrophil = st.number_input("Neutrophil (%)", min_value=0.0, max_value=100.0, value=78.0)
#     lymphocyte = st.number_input("Lymphocyte (%)", min_value=0.0, max_value=100.0, value=15.0)
#     rbc = st.number_input("RBC (10^6/μL)", min_value=0.0, max_value=10.0, value=3.8)

# with col4:
#     hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=20.0, value=10.5)
#     platelet = st.number_input("Platelet (10^3/μL)", min_value=0, max_value=1000, value=150)
#     ph = st.number_input("pH", min_value=6.5, max_value=8.0, value=7.3)
#     pain = st.number_input("Pain Score (0-10)", min_value=0, max_value=10, value=4)
#     rbs = st.number_input("RBS (mg/dL)", min_value=0, max_value=500, value=180)
#     oxygen = st.number_input("Oxygen Saturation (%)", min_value=0, max_value=100, value=92)

# # Create patient dictionary
# patient_data = {
#     'Center': center,
#     'sex': sex,
#     'age': age,
#     'address': address,
#     'payment_type': payment,
#     'DX_condition': diagnosis,
#     'Pulse_rate': pulse,
#     'SBP': sbp,
#     'DBP': dbp,
#     'Temp': temp,
#     'RR': rr,
#     'SPO2': spo2,
#     'Pain_score': pain,
#     'RBS': rbs,
#     'Oxygen': oxygen,
#     'WBC': wbc,
#     'Neutrophil': neutrophil,
#     'Lymphocyte': lymphocyte,
#     'RBC': rbc,
#     'Hemoglobin': hemoglobin,
#     'Platelet': platelet,
#     'PH': ph
# }

# st.markdown(
#     """
#     <style>
#     .reportview-container {
#         background-color: #f0f8ff; /* Change to your desired color */
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
# # Combined view
# if tool == "Combined View":
#     st.subheader("ICU Patient Risk Profile")
    
#     if st.button("Generate Comprehensive Risk Assessment"):
#         # Predict LOS first
#         los_prediction = los_predictor.predict(patient_data)
        
#         # Add LOS to patient data for mortality prediction
#         patient_data_with_los = patient_data.copy()
#         patient_data_with_los['LOS'] = los_prediction if los_prediction is not None else 3.0
        
#         # Predict mortality
#         mortality_pred, mortality_prob = mortality_predictor.predict(patient_data_with_los)
        
#         # Create dashboard columns
#         col1, col2 = st.columns(2)
        
#         # LOS Results
#         with col1:
#             st.subheader("Length of Stay Prediction")
#             if los_prediction is None:
#                 st.warning("⚠️ LOS Prediction Unavailable")
#                 los_prediction = 3.0  # Default value
#             elif los_prediction < 3:
#                 st.success(f"## Short Stay: {los_prediction} days")
#                 st.progress(30)
#                 st.markdown("""
#                 **Clinical Recommendations:**
#                 - Standard monitoring protocol
#                 - Daily reassessment
#                 - Early discharge planning
#                 - Patient education
#                 """)
#             elif los_prediction < 7:
#                 st.warning(f"## Moderate Stay: {los_prediction} days")
#                 st.progress(60)
#                 st.markdown("""
#                 **Clinical Recommendations:**
#                 - Enhanced monitoring
#                 - Multidisciplinary review
#                 - Respiratory therapy consult
#                 - Nutrition assessment
#                 """)
#             else:
#                 st.error(f"## Extended Stay: {los_prediction} days")
#                 st.progress(90)
#                 st.markdown("""
#                 **Clinical Recommendations:**
#                 - Critical care protocol
#                 - Ventilator reservation
#                 - Specialty consultation
#                 - Family conference
#                 - Nutrition support
#                 """)
            
#             # Resource planning
#             st.markdown("---")
#             st.subheader("Resource Planning")
#             if los_prediction > 7:
#                 st.warning("**Extended Stay Planning Required:**")
#                 st.markdown("- ICU bed for >1 week")
#                 st.markdown("- Ventilator availability check")
#                 st.markdown("- Specialty consultation needed")
#                 st.markdown("- Nutrition support team alert")
#             elif los_prediction > 3:
#                 st.info("**Moderate Resource Needs:**")
#                 st.markdown("- Intermediate care bed")
#                 st.markdown("- Physical therapy consult")
#                 st.markdown("- Daily lab monitoring")
#             else:
#                 st.success("**Standard Resources:**")
#                 st.markdown("- Routine ICU monitoring")
#                 st.markdown("- Discharge planning initiated")
        
#         # Mortality Results
#         with col2:
#             st.subheader("Mortality Risk Assessment")
#             if mortality_pred is None or mortality_prob is None:
#                 st.warning("⚠️ Mortality Prediction Unavailable")
#                 mortality_prob = 0.3  # Default value
#             elif mortality_pred == 1:  # High risk
#                 st.error(f"## ⚠️ High Mortality Risk: {mortality_prob:.1%}")
#                 st.progress(int(mortality_prob * 100))
#                 st.markdown("""
#                 **Clinical Recommendations:**
#                 - Immediate critical care protocol
#                 - Notify ICU attending
#                 - Goals of care discussion
#                 - Optimize hemodynamic support
#                 - Hourly vital monitoring
#                 """)
#             else:  # Low risk
#                 st.success(f"## ✅ Low Mortality Risk: {mortality_prob:.1%}")
#                 st.progress(int(mortality_prob * 100))
#                 st.markdown("""
#                 **Clinical Recommendations:**
#                 - Continue current management
#                 - Monitor for deterioration
#                 - Daily stability assessment
#                 - Step-down planning
#                 """)
            
#             # Risk factors
#             st.markdown("---")
#             st.subheader("Key Risk Factors")
#             risk_factors = []
#             if age > 65: risk_factors.append(f"- 👴 Advanced age ({age} years)")
#             if sbp < 90: risk_factors.append(f"- 💔 Hypotension (SBP: {sbp} mmHg)")
#             if spo2 < 90: risk_factors.append(f"- 😮‍💨 Hypoxia (SpO2: {spo2}%)")
#             if wbc > 15: risk_factors.append(f"- 🦠 Leukocytosis (WBC: {wbc} K/μL)")
#             if diagnosis == "Sepsis": risk_factors.append(f"- 🦠 Sepsis diagnosis")
#             if platelet < 100: risk_factors.append(f"- 🩸 Thrombocytopenia (Platelets: {platelet} K/μL)")
            
#             if risk_factors:
#                 st.warning("**Identified Risk Factors:**")
#                 for factor in risk_factors:
#                     st.markdown(factor)
#             else:
#                 st.success("No major risk factors identified")
        
#         # Summary panel at bottom
#         st.markdown("---")
#         st.subheader("Clinical Summary")
        
#         summary_col1, summary_col2, summary_col3 = st.columns(3)
        
#         with summary_col1:
#             if los_prediction is not None:
#                 st.metric("Predicted LOS", f"{los_prediction} days", 
#                           "Short stay" if los_prediction < 3 else 
#                           "Moderate stay" if los_prediction < 7 else "Extended stay")
#             else:
#                 st.metric("Predicted LOS", "Unavailable")
        
#         with summary_col2:
#             if mortality_prob is not None:
#                 st.metric("Mortality Risk", f"{mortality_prob:.1%}", 
#                           "Low risk" if mortality_prob < 0.3 else 
#                           "Moderate risk" if mortality_prob < 0.6 else "High risk",
#                           delta_color="inverse")
#             else:
#                 st.metric("Mortality Risk", "Unavailable")
        
#         with summary_col3:
#             if los_prediction is not None and mortality_prob is not None:
#                 acuity = "High Acuity" if mortality_prob > 0.6 or los_prediction > 7 else "Medium Acuity" if mortality_prob > 0.3 or los_prediction > 3 else "Low Acuity"
#                 st.metric("Patient Acuity", acuity)
#             else:
#                 st.metric("Patient Acuity", "Unavailable")
        
#         # Final recommendations
#         st.info("**Priority Actions:**")
#         if mortality_prob is not None and mortality_prob > 0.6:
#             st.markdown("- 🚨 Immediate attending physician notification")
#             st.markdown("- 🧑‍⚕️ Urgent multidisciplinary review")
#             st.markdown("- 👨‍👩‍👧‍👦 Family meeting within 24 hours")
#         elif los_prediction is not None and los_prediction > 7:
#             st.markdown("- 📋 Extended care protocol initiation")
#             st.markdown("- 🛌 Bed management notification")
#             st.markdown("- 🍲 Nutrition support consultation")
#         else:
#             st.markdown("- 📊 Routine monitoring plan")
#             st.markdown("- 📝 Daily progress assessment")
#             st.markdown("- 🏁 Discharge planning initiated")

# # Length of Stay tool
# elif tool == "Length of Stay":
#     st.subheader("ICU Length of Stay Predictor")
    
#     if st.button("Predict Length of Stay"):
#         los_prediction = los_predictor.predict(patient_data)
        
#         if los_prediction is None:
#             st.warning("⚠️ LOS Prediction Unavailable")
#             los_prediction = 3.0  # Default value
            
#         # Results
#         col1, col2 = st.columns([1, 2])
        
#         with col1:
#             if los_prediction < 3:
#                 st.success(f"## Short Stay: {los_prediction} days")
#                 st.progress(30)
#             elif los_prediction < 7:
#                 st.warning(f"## Moderate Stay: {los_prediction} days")
#                 st.progress(60)
#             else:
#                 st.error(f"## Extended Stay: {los_prediction} days")
#                 st.progress(90)
            
#             st.metric("Predicted ICU Days", los_prediction)
        
#         with col2:
#             st.subheader("Clinical Recommendations")
#             if los_prediction < 3:
#                 st.markdown("""
#                 **Care Plan:**
#                 - Standard monitoring protocol
#                 - Daily physician assessment
#                 - Discharge planning initiated
#                 - Patient education materials
                
#                 **Resource Planning:**
#                 - Routine nursing care
#                 - Standard monitoring equipment
#                 - Discharge coordination
#                 """)
#             elif los_prediction < 7:
#                 st.markdown("""
#                 **Care Plan:**
#                 - Enhanced vital monitoring
#                 - Multidisciplinary team review
#                 - Respiratory therapy consult
#                 - Nutrition assessment
                
#                 **Resource Planning:**
#                 - Intermediate care bed
#                 - Physical therapy consult
#                 - Daily lab monitoring
#                 - Social work assessment
#                 """)
#             else:
#                 st.markdown("""
#                 **Care Plan:**
#                 - Critical care protocol
#                 - Ventilator management plan
#                 - Specialty consultations
#                 - Family conference
#                 - Aggressive nutrition support
                
#                 **Resource Planning:**
#                 - ICU bed reservation >1 week
#                 - Ventilator availability confirmation
#                 - Specialty team notifications
#                 - Long-term medication planning
#                 """)
        
#         # Risk factors
#         st.markdown("---")
#         st.subheader("Extended Stay Risk Factors")
#         risk_col1, risk_col2, risk_col3 = st.columns(3)
        
#         with risk_col1:
#             if age > 65: st.error(f"👴 Advanced age ({age} years)")
#             if diagnosis == "Sepsis": st.error("🦠 Sepsis diagnosis")
        
#         with risk_col2:
#             if spo2 < 90: st.error(f"😮‍💨 Hypoxia (SpO2: {spo2}%)")
#             if platelet < 100: st.error(f"🩸 Thrombocytopenia")
        
#         with risk_col3:
#             if wbc > 15: st.error(f"🦠 Leukocytosis (WBC: {wbc})")
#             if hemoglobin < 10: st.error(f"🩸 Anemia (Hgb: {hemoglobin})")

# # Mortality Risk tool
# elif tool == "Mortality Risk":
#     st.subheader("ICU Mortality Risk Predictor")
    
#     # Add LOS input specifically for mortality prediction
#     los_days = st.number_input("Current Length of Stay (days)", min_value=0, max_value=100, value=3)
#     patient_data['LOS'] = los_days
    
#     if st.button("Predict Mortality Risk"):
#         mortality_pred, mortality_prob = mortality_predictor.predict(patient_data)
        
#         if mortality_pred is None or mortality_prob is None:
#             st.warning("⚠️ Mortality Prediction Unavailable")
#             mortality_prob = 0.3  # Default value
            
#         # Results
#         col1, col2 = st.columns([1, 2])
        
#         with col1:
#             if mortality_pred == 1:
#                 st.error(f"## ⚠️ High Risk: {mortality_prob:.1%}")
#                 st.progress(int(mortality_prob * 100))
#             else:
#                 st.success(f"## ✅ Low Risk: {mortality_prob:.1%}")
#                 st.progress(int(mortality_prob * 100))
            
#             st.metric("Mortality Probability", f"{mortality_prob:.1%}")
        
#         with col2:
#             st.subheader("Clinical Recommendations")
#             if mortality_pred == 1:
#                 st.markdown("""
#                 **Immediate Actions:**
#                 - Initiate critical care protocol
#                 - Notify ICU attending physician
#                 - Optimize hemodynamic support
#                 - Consider vasopressor support
#                 - Hourly vital sign monitoring
                
#                 **Communication Plan:**
#                 - Family meeting within 24 hours
#                 - Goals of care discussion
#                 - Code status clarification
#                 """)
#             else:
#                 st.markdown("""
#                 **Management Plan:**
#                 - Continue current treatment
#                 - Monitor for deterioration
#                 - Daily stability assessment
#                 - Step-down unit planning
                
#                 **Prevention Strategies:**
#                 - VTE prophylaxis
#                 - Stress ulcer prevention
#                 - Daily sedation vacation
#                 - Early mobilization
#                 """)
        
#         # Risk factors
#         st.markdown("---")
#         st.subheader("Key Risk Factors")
#         risk_col1, risk_col2, risk_col3 = st.columns(3)
        
#         with risk_col1:
#             if age > 65: st.error(f"👴 Advanced age ({age} years)")
#             if sbp < 90: st.error(f"💔 Hypotension (SBP: {sbp} mmHg)")
        
#         with risk_col2:
#             if spo2 < 90: st.error(f"😮‍💨 Hypoxia (SpO2: {spo2}%)")
#             if rbs > 200: st.error(f"🍬 Hyperglycemia (RBS: {rbs})")
        
#         with risk_col3:
#             if diagnosis == "Sepsis": st.error(f"🦠 Sepsis diagnosis")
#             if los_days > 7: st.error(f"📅 Extended LOS ({los_days} days)")

# # Add footer
# st.markdown("---")
# st.caption("ICU Clinical Predictor Suite v2.0 | For clinical use only | Maintain patient confidentiality")
# st.caption("Predictions are estimates based on historical data. Actual outcomes may vary.")
# ICU_los_Mortality_Streamlit_App.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.compose import ColumnTransformer as RealColumnTransformer
from sklearn.metrics import DistanceMetric as RealDistanceMetric

# =============================================================================
# CUSTOM STYLING FOR ETHIOPIAN ICU CONTEXT
# =============================================================================
st.set_page_config(
    page_title="ICU Clinical Predictor Suite - Ethiopia",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
st.markdown("""
<style>
    /* Base styling */
    :root {
        --eth-green: #078930;
        --eth-yellow: #FCDD09;
        --eth-red: #DA121A;
        --eth-blue: #0D47A1;
        --light-bg: #f8f9fa;
        --card-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    
    /* Main layout */
    .stApp {
        background-color: #f0f8ff;
        background-image: linear-gradient(to bottom, #e6f7ff, #ffffff);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, var(--eth-green), #0a6e28) !important;
        color: white !important;
        padding: 1rem;
    }
    
    .sidebar .sidebar-content {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {
        color: white !important;
    }
    
    /* Header styling */
    header[data-testid="stHeader"] {
        background: linear-gradient(to right, var(--eth-green), var(--eth-blue)) !important;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: var(--card-shadow);
        border-left: 4px solid var(--eth-blue);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    .card-critical {
        border-left: 4px solid var(--eth-red);
    }
    
    .card-warning {
        border-left: 4px solid var(--eth-yellow);
    }
    
    .card-success {
        border-left: 4px solid var(--eth-green);
    }
    
    /* Button styling */
    .stButton>button {
        background: var(--eth-blue) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        background: #0b3d91 !important;
        transform: scale(1.05);
    }
    
    /* Metric styling */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: var(--card-shadow);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(to right, var(--eth-green), var(--eth-yellow), var(--eth-red)) !important;
    }
    
    /* Icon styling */
    .icon-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: var(--eth-blue);
        color: white;
        margin-right: 10px;
        font-size: 20px;
    }
    
    /* Form styling */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        border-radius: 8px !important;
        border: 1px solid #ddd !important;
    }
    
    /* Responsive layout */
    @media (max-width: 768px) {
        .stColumn {
            margin-bottom: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PREDICTOR CLASSES
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
# STREAMLIT APP LAYOUT - ENHANCED FOR ETHIOPIAN CONTEXT
# =============================================================================

# Main header with Ethiopian flag colors
st.markdown("""
<div style="background: linear-gradient(to right, #078930, #FCDD09, #DA121A, #0D47A1); 
            height: 8px; border-radius: 4px; margin-bottom: 1rem;"></div>
""", unsafe_allow_html=True)

st.title('🏥 ICU Clinical Predictor Suite - Ethiopia')

# Subheader with contextual information
st.markdown("""
<p style="font-size: 1.1rem; color: #333; border-bottom: 2px solid #0D47A1; 
            padding-bottom: 0.5rem; margin-bottom: 1.5rem;">
Clinical Decision Support for Ethiopian ICU Professionals
</p>
""", unsafe_allow_html=True)

# Sidebar with tool selection and information - Enhanced
with st.sidebar:
    st.header("Prediction Tools")
    tool = st.radio("Select Tool:", ["Combined View", "Length of Stay", "Mortality Risk"])
    
    st.header("About This Suite")
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
        <p style="color: white; margin-bottom: 0.5rem;"><b>Integrated ICU Prediction Tools:</b></p>
        <ul style="color: white; padding-left: 1.5rem;">
            <li>Length of Stay (LOS) Prediction</li>
            <li>Mortality Risk Assessment</li>
            <li>Resource Planning Guidance</li>
            <li>Clinical Decision Support</li>
        </ul>
        
        <p style="color: white; margin-top: 1rem; margin-bottom: 0.5rem;"><b>Data Sources:</b></p>
        <ul style="color: white; padding-left: 1.5rem;">
            <li>10,810+ ICU patient records</li>
            <li>25+ clinical parameters</li>
            <li>Ethiopian ICU Registry (2022-2025)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header("System Status")
    status_container = st.container()
    
    st.markdown("---")
    
    st.header("Important Notes")
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
        <ul style="color: white; padding-left: 1.5rem;">
            <li>For clinical use only</li>
            <li>Maintain patient confidentiality</li>
            <li>Predictions are estimates</li>
            <li>Always combine with clinical judgment</li>
            <li>Reassess patients regularly</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<p style="color: white; text-align: center;">ICU Clinical Predictor Suite v2.0</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: rgba(255,255,255,0.8); text-align: center; font-size: 0.9rem;">Developed for Ethiopian Healthcare</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: rgba(255,255,255,0.7); text-align: center; font-size: 0.8rem;">Last updated: June 7, 2025</p>', unsafe_allow_html=True)

# Update system status in sidebar
with status_container:
    if los_predictor.model and mortality_predictor.model:
        st.success("✅ All models operational", icon="✅")
    else:
        st.warning("⚠️ Fallback models in use - some features limited", icon="⚠️")

# Patient form - Enhanced with cards and better organization
st.header("📋 Patient Information")

# Form layout in two columns
col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("Demographics")
        center = st.selectbox("Medical Center", ["Yekatit12","Minilik","RASDESTA","ZEWUDITU","DBU", "St. Peter", "Black Lion", "Tikur Anbessa", "Zewditu"])
        sex = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=0, max_value=120, value=58)
        address = st.selectbox("Address", ["Addis Ababa", "Outside Addis", "Rural"])
        payment = st.selectbox("Payment Type", ["CBHI", "Private", "Government", "Charity", "Unknown"])
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("Clinical Presentation")
        diagnosis = st.selectbox("Primary Diagnosis", [
            "Cardiovascular", "Respiratory", "Sepsis", "Trauma", 
            "Neurological", "Gastrointestinal", "Renal", "Metabolic"
        ])
        pulse = st.number_input("Pulse Rate (bpm)", min_value=0, max_value=200, value=95)
        sbp = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=300, value=110)
        dbp = st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=200, value=65)
        temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.8)
        rr = st.number_input("Respiratory Rate", min_value=0, max_value=60, value=22)
        spo2 = st.number_input("SPO2 (%)", min_value=0, max_value=100, value=92)
        st.markdown('</div>', unsafe_allow_html=True)

# Lab values in a separate section with two columns
st.header("🔬 Laboratory Values")
col3, col4 = st.columns(2)

with col3:
    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        wbc = st.number_input("WBC (10^3/μL)", min_value=0.0, max_value=100.0, value=12.5)
        neutrophil = st.number_input("Neutrophil (%)", min_value=0.0, max_value=100.0, value=78.0)
        lymphocyte = st.number_input("Lymphocyte (%)", min_value=0.0, max_value=100.0, value=15.0)
        rbc = st.number_input("RBC (10^6/μL)", min_value=0.0, max_value=10.0, value=3.8)
        st.markdown('</div>', unsafe_allow_html=True)

with col4:
    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=20.0, value=10.5)
        platelet = st.number_input("Platelet (10^3/μL)", min_value=0, max_value=1000, value=150)
        ph = st.number_input("pH", min_value=6.5, max_value=8.0, value=7.3)
        pain = st.number_input("Pain Score (0-10)", min_value=0, max_value=10, value=4)
        rbs = st.number_input("RBS (mg/dL)", min_value=0, max_value=500, value=180)
        oxygen = st.number_input("Oxygen Saturation (%)", min_value=0, max_value=100, value=92)
        st.markdown('</div>', unsafe_allow_html=True)

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

# Combined view - Enhanced with visual hierarchy
if tool == "Combined View":
    st.header("📊 Comprehensive Patient Assessment")
    
    if st.button("Generate Risk Assessment", key="combined_btn", help="Predict both LOS and Mortality Risk"):
        # Predict LOS first
        with st.spinner("Calculating Length of Stay..."):
            los_prediction = los_predictor.predict(patient_data)
        
        # Add LOS to patient data for mortality prediction
        patient_data_with_los = patient_data.copy()
        patient_data_with_los['LOS'] = los_prediction if los_prediction is not None else 3.0
        
        # Predict mortality
        with st.spinner("Assessing Mortality Risk..."):
            mortality_pred, mortality_prob = mortality_predictor.predict(patient_data_with_los)
        
        # Create dashboard columns
        col1, col2 = st.columns(2)
        
        # LOS Results
        with col1:
            card_class = "custom-card "
            if los_prediction is None:
                st.warning("⚠️ LOS Prediction Unavailable")
                los_prediction = 3.0  # Default value
                card_class += "card-warning"
            elif los_prediction < 3:
                card_class += "card-success"
            elif los_prediction < 7:
                card_class += "card-warning"
            else:
                card_class += "card-critical"
                
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            st.subheader("📅 Length of Stay Prediction")
            
            if los_prediction < 3:
                st.markdown(f'<h2 style="color: #078930;">Short Stay: {los_prediction} days</h2>', unsafe_allow_html=True)
                st.progress(30)
            elif los_prediction < 7:
                st.markdown(f'<h2 style="color: #FCDD09;">Moderate Stay: {los_prediction} days</h2>', unsafe_allow_html=True)
                st.progress(60)
            else:
                st.markdown(f'<h2 style="color: #DA121A;">Extended Stay: {los_prediction} days</h2>', unsafe_allow_html=True)
                st.progress(90)
            
            # Resource planning
            st.subheader("🛠 Resource Planning")
            if los_prediction > 7:
                st.markdown("""
                <div style="background: #FFF5F5; padding: 1rem; border-radius: 8px; border-left: 3px solid #DA121A;">
                    <p><b>Extended Stay Planning Required:</b></p>
                    <ul>
                        <li>ICU bed for >1 week</li>
                        <li>Ventilator availability check</li>
                        <li>Specialty consultation needed</li>
                        <li>Nutrition support team alert</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif los_prediction > 3:
                st.markdown("""
                <div style="background: #FFFBF0; padding: 1rem; border-radius: 8px; border-left: 3px solid #FCDD09;">
                    <p><b>Moderate Resource Needs:</b></p>
                    <ul>
                        <li>Intermediate care bed</li>
                        <li>Physical therapy consult</li>
                        <li>Daily lab monitoring</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #F5FFF7; padding: 1rem; border-radius: 8px; border-left: 3px solid #078930;">
                    <p><b>Standard Resources:</b></p>
                    <ul>
                        <li>Routine ICU monitoring</li>
                        <li>Discharge planning initiated</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Mortality Results
        with col2:
            card_class = "custom-card "
            if mortality_pred is None or mortality_prob is None:
                st.warning("⚠️ Mortality Prediction Unavailable")
                mortality_prob = 0.3  # Default value
                card_class += "card-warning"
            elif mortality_pred == 1:  # High risk
                card_class += "card-critical"
            else:  # Low risk
                card_class += "card-success"
                
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            st.subheader("⚠️ Mortality Risk Assessment")
            
            if mortality_pred == 1:  # High risk
                st.markdown(f'<h2 style="color: #DA121A;">High Mortality Risk: {mortality_prob:.1%}</h2>', unsafe_allow_html=True)
                st.progress(int(mortality_prob * 100))
            else:  # Low risk
                st.markdown(f'<h2 style="color: #078930;">Low Mortality Risk: {mortality_prob:.1%}</h2>', unsafe_allow_html=True)
                st.progress(int(mortality_prob * 100))
            
            # Risk factors
            st.subheader("🔍 Key Risk Factors")
            risk_factors = []
            if age > 65: risk_factors.append(f"👴 Advanced age ({age} years)")
            if sbp < 90: risk_factors.append(f"💔 Hypotension (SBP: {sbp} mmHg)")
            if spo2 < 90: risk_factors.append(f"😮‍💨 Hypoxia (SpO2: {spo2}%)")
            if wbc > 15: risk_factors.append(f"🦠 Leukocytosis (WBC: {wbc} K/μL)")
            if diagnosis == "Sepsis": risk_factors.append(f"🦠 Sepsis diagnosis")
            if platelet < 100: risk_factors.append(f"🩸 Thrombocytopenia (Platelets: {platelet} K/μL)")
            
            if risk_factors:
                risk_html = '<div style="background: #FFF5F5; padding: 1rem; border-radius: 8px;">'
                risk_html += '<p><b>Identified Risk Factors:</b></p><div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;">'
                for factor in risk_factors:
                    risk_html += f'<div style="background: #FFEBEE; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">{factor}</div>'
                risk_html += '</div></div>'
                st.markdown(risk_html, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #F5FFF7; padding: 1rem; border-radius: 8px;">
                    <p><b>No major risk factors identified</b></p>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Summary panel at bottom
        st.markdown("---")
        st.subheader("📋 Clinical Summary")
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            if los_prediction is not None:
                st.metric("Predicted LOS", f"{los_prediction} days", 
                          "Short stay" if los_prediction < 3 else 
                          "Moderate stay" if los_prediction < 7 else "Extended stay",
                          delta_color="normal")
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
        st.markdown("---")
        st.subheader("🚀 Priority Actions")
        if mortality_prob is not None and mortality_prob > 0.6:
            st.markdown("""
            <div style="background: #FFF5F5; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #DA121A;">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <span class="icon-badge">🚨</span>
                    <h3>Critical Care Priority Actions</h3>
                </div>
                <ul>
                    <li>Immediate attending physician notification</li>
                    <li>Urgent multidisciplinary review</li>
                    <li>Family meeting within 24 hours</li>
                    <li>Continuous hemodynamic monitoring</li>
                    <li>Prepare for possible escalation of care</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif los_prediction is not None and los_prediction > 7:
            st.markdown("""
            <div style="background: #FFFBF0; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #FCDD09;">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <span class="icon-badge">📋</span>
                    <h3>Extended Care Management</h3>
                </div>
                <ul>
                    <li>Extended care protocol initiation</li>
                    <li>Bed management notification</li>
                    <li>Nutrition support consultation</li>
                    <li>Pressure ulcer prevention plan</li>
                    <li>Psychological support assessment</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #F5FFF7; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #078930;">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <span class="icon-badge">📊</span>
                    <h3>Standard Monitoring Protocol</h3>
                </div>
                <ul>
                    <li>Routine monitoring plan</li>
                    <li>Daily progress assessment</li>
                    <li>Discharge planning initiated</li>
                    <li>Patient education materials</li>
                    <li>Family communication update</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Length of Stay tool - Enhanced
elif tool == "Length of Stay":
    st.header("📅 ICU Length of Stay Predictor")
    
    if st.button("Predict Length of Stay", key="los_btn"):
        with st.spinner("Calculating predicted stay duration..."):
            los_prediction = los_predictor.predict(patient_data)
        
        if los_prediction is None:
            st.warning("⚠️ LOS Prediction Unavailable")
            los_prediction = 3.0  # Default value
            
        # Results in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            if los_prediction < 3:
                st.markdown(f'<h2 style="color: #078930;">Short Stay: {los_prediction} days</h2>', unsafe_allow_html=True)
                st.progress(30)
            elif los_prediction < 7:
                st.markdown(f'<h2 style="color: #FCDD09;">Moderate Stay: {los_prediction} days</h2>', unsafe_allow_html=True)
                st.progress(60)
            else:
                st.markdown(f'<h2 style="color: #DA121A;">Extended Stay: {los_prediction} days</h2>', unsafe_allow_html=True)
                st.progress(90)
            
            st.metric("Predicted ICU Days", los_prediction)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Risk factors
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("⚠️ Extended Stay Risk Factors")
            risk_factors = []
            if age > 65: risk_factors.append(f"👴 Advanced age ({age} years)")
            if diagnosis == "Sepsis": risk_factors.append("🦠 Sepsis diagnosis")
            if spo2 < 90: risk_factors.append(f"😮‍💨 Hypoxia (SpO2: {spo2}%)")
            if platelet < 100: risk_factors.append("🩸 Thrombocytopenia")
            if wbc > 15: risk_factors.append(f"🦠 Leukocytosis (WBC: {wbc})")
            if hemoglobin < 10: risk_factors.append(f"🩸 Anemia (Hgb: {hemoglobin})")
            
            if risk_factors:
                st.markdown('<div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;">', unsafe_allow_html=True)
                for factor in risk_factors:
                    st.markdown(f'<div style="background: #FFF5F5; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">{factor}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No significant risk factors for extended stay identified")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("📋 Clinical Recommendations")
            if los_prediction < 3:
                st.markdown("""
                <div style="background: #F5FFF7; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    <p><b>Care Plan:</b></p>
                    <ul>
                        <li>Standard monitoring protocol</li>
                        <li>Daily physician assessment</li>
                        <li>Discharge planning initiated</li>
                        <li>Patient education materials</li>
                    </ul>
                </div>
                <div style="background: #F0F9FF; padding: 1rem; border-radius: 8px;">
                    <p><b>Resource Planning:</b></p>
                    <ul>
                        <li>Routine nursing care</li>
                        <li>Standard monitoring equipment</li>
                        <li>Discharge coordination</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif los_prediction < 7:
                st.markdown("""
                <div style="background: #FFFBF0; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    <p><b>Care Plan:</b></p>
                    <ul>
                        <li>Enhanced vital monitoring</li>
                        <li>Multidisciplinary team review</li>
                        <li>Respiratory therapy consult</li>
                        <li>Nutrition assessment</li>
                    </ul>
                </div>
                <div style="background: #F0F9FF; padding: 1rem; border-radius: 8px;">
                    <p><b>Resource Planning:</b></p>
                    <ul>
                        <li>Intermediate care bed</li>
                        <li>Physical therapy consult</li>
                        <li>Daily lab monitoring</li>
                        <li>Social work assessment</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #FFF5F5; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    <p><b>Care Plan:</b></p>
                    <ul>
                        <li>Critical care protocol</li>
                        <li>Ventilator management plan</li>
                        <li>Specialty consultations</li>
                        <li>Family conference</li>
                        <li>Aggressive nutrition support</li>
                    </ul>
                </div>
                <div style="background: #F0F9FF; padding: 1rem; border-radius: 8px;">
                    <p><b>Resource Planning:</b></p>
                    <ul>
                        <li>ICU bed reservation >1 week</li>
                        <li>Ventilator availability confirmation</li>
                        <li>Specialty team notifications</li>
                        <li>Long-term medication planning</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# Mortality Risk tool - Enhanced
elif tool == "Mortality Risk":
    st.header("⚠️ ICU Mortality Risk Predictor")
    
    # Add LOS input specifically for mortality prediction
    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        los_days = st.number_input("Current Length of Stay (days)", min_value=0, max_value=100, value=3)
        patient_data['LOS'] = los_days
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Predict Mortality Risk", key="mortality_btn"):
        with st.spinner("Assessing mortality risk..."):
            mortality_pred, mortality_prob = mortality_predictor.predict(patient_data)
        
        if mortality_pred is None or mortality_prob is None:
            st.warning("⚠️ Mortality Prediction Unavailable")
            mortality_prob = 0.3  # Default value
            
        # Results in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            if mortality_pred == 1:
                st.markdown(f'<h2 style="color: #DA121A;">High Risk: {mortality_prob:.1%}</h2>', unsafe_allow_html=True)
                st.progress(int(mortality_prob * 100))
            else:
                st.markdown(f'<h2 style="color: #078930;">Low Risk: {mortality_prob:.1%}</h2>', unsafe_allow_html=True)
                st.progress(int(mortality_prob * 100))
            
            st.metric("Mortality Probability", f"{mortality_prob:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Risk factors
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("🔍 Key Risk Factors")
            risk_factors = []
            if age > 65: risk_factors.append(f"👴 Advanced age ({age} years)")
            if sbp < 90: risk_factors.append(f"💔 Hypotension (SBP: {sbp} mmHg)")
            if spo2 < 90: risk_factors.append(f"😮‍💨 Hypoxia (SpO2: {spo2}%)")
            if rbs > 200: risk_factors.append(f"🍬 Hyperglycemia (RBS: {rbs})")
            if diagnosis == "Sepsis": risk_factors.append("🦠 Sepsis diagnosis")
            if los_days > 7: risk_factors.append(f"📅 Extended LOS ({los_days} days)")
            
            if risk_factors:
                st.markdown('<div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;">', unsafe_allow_html=True)
                for factor in risk_factors:
                    st.markdown(f'<div style="background: #FFF5F5; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">{factor}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No major mortality risk factors identified")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("📋 Clinical Recommendations")
            if mortality_pred == 1:
                st.markdown("""
                <div style="background: #FFF5F5; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    <p><b>Immediate Actions:</b></p>
                    <ul>
                        <li>Initiate critical care protocol</li>
                        <li>Notify ICU attending physician</li>
                        <li>Optimize hemodynamic support</li>
                        <li>Consider vasopressor support</li>
                        <li>Hourly vital sign monitoring</li>
                    </ul>
                </div>
                <div style="background: #F0F9FF; padding: 1rem; border-radius: 8px;">
                    <p><b>Communication Plan:</b></p>
                    <ul>
                        <li>Family meeting within 24 hours</li>
                        <li>Goals of care discussion</li>
                        <li>Code status clarification</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #F5FFF7; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    <p><b>Management Plan:</b></p>
                    <ul>
                        <li>Continue current treatment</li>
                        <li>Monitor for deterioration</li>
                        <li>Daily stability assessment</li>
                        <li>Step-down unit planning</li>
                    </ul>
                </div>
                <div style="background: #F0F9FF; padding: 1rem; border-radius: 8px;">
                    <p><b>Prevention Strategies:</b></p>
                    <ul>
                        <li>VTE prophylaxis</li>
                        <li>Stress ulcer prevention</li>
                        <li>Daily sedation vacation</li>
                        <li>Early mobilization</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# Add footer with Ethiopian healthcare context
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; background: #f0f8ff; border-radius: 8px;">
    <p style="margin-bottom: 0.5rem; font-weight: 600;">ICU Clinical Predictor Suite v2.0 | For clinical use only | Maintain patient confidentiality</p>
    <p style="font-size: 0.9rem; color: #555;">Developed for Ethiopian Healthcare System | Predictions are estimates based on historical Ethiopian ICU data</p>
</div>
""", unsafe_allow_html=True)