import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Title
st.title("💼 Employee Salary Prediction App")

# Load data
data = pd.read_csv("ds_salaries.csv")

# Keep required columns
data = data[
    [
        'experience_level',
        'employment_type',
        'job_title',
        'employee_residence',
        'remote_ratio',
        'company_location',
        'company_size',
        'salary_in_usd'
    ]
]

# Drop missing values
data = data.dropna()

# Store encoders
encoders = {}

# Encode categorical columns
categorical_cols = [
    'experience_level',
    'employment_type',
    'job_title',
    'employee_residence',
    'company_location',
    'company_size'
]

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

# Split data
X = data.drop("salary_in_usd", axis=1)
y = data["salary_in_usd"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------- UI ---------------- #

st.subheader("Enter Employee Details")

experience = st.selectbox("Experience Level", encoders['experience_level'].classes_)
employment = st.selectbox("Employment Type", encoders['employment_type'].classes_)
job = st.selectbox("Job Title", encoders['job_title'].classes_)
residence = st.selectbox("Employee Residence", encoders['employee_residence'].classes_)
remote = st.slider("Remote Ratio", 0, 100, 50)
company_location = st.selectbox("Company Location", encoders['company_location'].classes_)
company_size = st.selectbox("Company Size", encoders['company_size'].classes_)

# Convert input using SAME encoders
input_data = pd.DataFrame({
    "experience_level": [encoders['experience_level'].transform([experience])[0]],
    "employment_type": [encoders['employment_type'].transform([employment])[0]],
    "job_title": [encoders['job_title'].transform([job])[0]],
    "employee_residence": [encoders['employee_residence'].transform([residence])[0]],
    "remote_ratio": [remote],
    "company_location": [encoders['company_location'].transform([company_location])[0]],
    "company_size": [encoders['company_size'].transform([company_size])[0]],
})

# Predict
if st.button("Predict Salary"):
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted Salary: $ {prediction[0]:,.2f}")