
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load model from Hugging Face Model Hub

model_path = hf_hub_download(
    repo_id="ajeet04yadav/tourism-package-model",
    filename="best_tourism_package_model.joblib"
)

model = joblib.load(model_path)

# Streamlit UI

st.title("Wellness Tourism Package Prediction App")

st.write("""
This application predicts whether a customer is likely to purchase the Wellness Tourism Package.
Please enter the customer details below to get a prediction.
""")

# User Inputs

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

typeofcontact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

citytier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

numberofpersonvisiting = st.number_input(
    "Number Of Persons Visiting",
    min_value=1,
    max_value=20,
    value=2
)

preferredpropertystar = st.selectbox(
    "Preferred Property Star",
    [1, 2, 3, 4, 5]
)

maritalstatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

numberoftrips = st.number_input(
    "Number Of Trips",
    min_value=0,
    max_value=50,
    value=3
)

passport = st.selectbox(
    "Passport",
    [0, 1]
)

owncar = st.selectbox(
    "Own Car",
    [0, 1]
)

numberofchildrenvisiting = st.number_input(
    "Number Of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)

designation = st.text_input(
    "Designation",
    value="Executive"
)

monthlyincome = st.number_input(
    "Monthly Income",
    min_value=1000,
    value=30000
)

pitchsatisfactionscore = st.slider(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

productpitched = st.text_input(
    "Product Pitched",
    value="Basic"
)

numberoffollowups = st.number_input(
    "Number Of Followups",
    min_value=0,
    max_value=20,
    value=2
)

durationofpitch = st.number_input(
    "Duration Of Pitch",
    min_value=0,
    max_value=300,
    value=20
)

# Assemble Input DataFrame

input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeofcontact,
    'CityTier': citytier,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': numberofpersonvisiting,
    'PreferredPropertyStar': preferredpropertystar,
    'MaritalStatus': maritalstatus,
    'NumberOfTrips': numberoftrips,
    'Passport': passport,
    'OwnCar': owncar,
    'NumberOfChildrenVisiting': numberofchildrenvisiting,
    'Designation': designation,
    'MonthlyIncome': monthlyincome,
    'PitchSatisfactionScore': pitchsatisfactionscore,
    'ProductPitched': productpitched,
    'NumberOfFollowups': numberoffollowups,
    'DurationOfPitch': durationofpitch
}])

# Prediction

if st.button("Predict Purchase"):

    prediction = model.predict(input_data)[0]

    result = (
        "Customer Likely To Purchase Package"
        if prediction == 1
        else
        "Customer Unlikely To Purchase Package"
    )

    st.subheader("Prediction Result")

    st.success(result)
