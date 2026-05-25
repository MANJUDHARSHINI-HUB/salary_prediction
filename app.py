import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Title
st.title("💼 Employee Salary Prediction")

# Load Dataset
data = pd.read_csv("ds_salaries.csv")

# Select Columns
data = data[[
    'experience_level',
    'employment_type',
    'job_title',
    'employee_residence',
    'remote_ratio',
    'company_location',
    'company_size',
    'salary_in_usd'
]]

# Remove Missing Values
data = data.dropna()

# Encode Text Columns
label_encoders = {}

for column in data.columns:

    if data[column].dtype == 'object':

        le = LabelEncoder()

        data[column] = le.fit_transform(
            data[column].astype(str)
        )

        label_encoders[column] = le

# Features and Target
X = data.drop('salary_in_usd', axis=1)
y = data['salary_in_usd']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# User Inputs
st.subheader("Enter Employee Details")

experience = st.selectbox(
    "Experience Level",
    ['EN', 'MI', 'SE', 'EX']
)

employment = st.selectbox(
    "Employment Type",
    ['FT', 'PT', 'CT', 'FL']
)

job = st.selectbox(
    "Job Title",
    data['job_title'].unique()
)

residence = st.selectbox(
    "Employee Residence",
    ['US', 'IN', 'GB', 'CA']
)

remote = st.slider(
    "Remote Ratio",
    0,
    100,
    50
)

company_location = st.selectbox(
    "Company Location",
    ['US', 'IN', 'GB', 'CA']
)

company_size = st.selectbox(
    "Company Size",
    ['S', 'M', 'L']
)

# Encode User Inputs
experience_encoded = label_encoders[
    'experience_level'
].transform([experience])[0]

employment_encoded = label_encoders[
    'employment_type'
].transform([employment])[0]

# Fix Job Titles
job_names = label_encoders['job_title'].classes_

selected_job = st.selectbox(
    "Choose Job Role",
    job_names
)

job_encoded = label_encoders[
    'job_title'
].transform([selected_job])[0]

residence_encoded = label_encoders[
    'employee_residence'
].transform([residence])[0]

location_encoded = label_encoders[
    'company_location'
].transform([company_location])[0]

size_encoded = label_encoders[
    'company_size'
].transform([company_size])[0]

# Prediction
if st.button("Predict Salary"):

    input_data = pd.DataFrame({
        'experience_level': [experience_encoded],
        'employment_type': [employment_encoded],
        'job_title': [job_encoded],
        'employee_residence': [residence_encoded],
        'remote_ratio': [remote],
        'company_location': [location_encoded],
        'company_size': [size_encoded]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Salary: $ {prediction[0]:,.2f}"
    )