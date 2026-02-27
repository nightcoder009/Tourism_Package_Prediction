import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="arijitchaki009/Tourism-Package-Prediction", filename="tourism_customer_purchase_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Package Purchase Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase a new tourism package
based on their demographic and interaction details.
""")
st.write("Please enter the customer information below to get the prediction.")

# Collect user input
# Numerical Inputs
Age = st.number_input("Age (Customer age in years)", min_value=18, max_value=100, value=30)
CityTier = st.selectbox("City Tier (Tier 1 = Metro, Tier 2 = Developed, Tier 3 = Smaller city)",[1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (Duration of sales pitch in minutes)",min_value=0,value=200)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting (Total people traveling together)",min_value=1,value=20)
NumberOfFollowups = st.number_input("Number of Follow-ups (Total follow-ups done by salesperson)",min_value=0,value=20)
PreferredPropertyStar = st.selectbox("Preferred Property Star (Preferred hotel rating)",[1, 2, 3, 4, 5])
NumberOfTrips = st.number_input("Number of Trips per Year",min_value=0,value=100)
Passport = st.selectbox("Passport Availability",["Yes", "No"])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score (1 = Low, 5 = High)",1, 5, 3)
OwnCar = st.selectbox("Owns a Car?",["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (Below age 5)",min_value=0,value=10)
MonthlyIncome = st.number_input("Monthly Income",min_value=0.0,value=300000.0)

# Categorical Inputs
TypeofContact = st.selectbox("Type of Contact",["Company Invited", "Self Inquiry"])
Occupation = st.selectbox("Occupation",["Salaried", "Freelancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender",["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status",["Single", "Married", "Divorced", "Unmarried"])
Designation = st.selectbox("Designation",["Executive", "Manager", "Senior Manager", "VP", "AVP"])


# Create Input DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation
}])

# Set the classification threshold
classification_threshold = 0.5

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase" if prediction == 1 else "not purchase"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
