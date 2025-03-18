import streamlit as st
import pandas as pd
import joblib

# Load models
model_paths = {
    'kmeans': 'models/kmeans_model.pkl',
    'scaler': 'models/scaler.pkl',
    'multi_target_classifier': 'models/multi_target_classifier_model.pkl'
}

kmeans = joblib.load(model_paths['kmeans'])
scaler = joblib.load(model_paths['scaler'])
multi_target_classifier = joblib.load(model_paths['multi_target_classifier'])

st.title("Customer Segmentation & Product Recommendation")

# User Input Fields
customer_id = st.text_input("Customer ID", "2a0b0e52-1df5-495c-b5a5-baca9198f32d")
name = st.text_input("Name", "Heera")
age = st.number_input("Age", min_value=18, max_value=100, value=35)
gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
marital_status = st.selectbox("Marital Status", ["Single", "Married"], index=0)
occupation = st.text_input("Occupation", "Industrial buyer")
max_dpd = st.number_input("Max DPD", value=30.0)
default_status = st.selectbox("Default Status", [0.0, 1.0], index=1)
transaction_amount = st.number_input("Transaction Amount", value=57500)
account_balance = st.number_input("Account Balance", value=145600)
is_salary = st.selectbox("Is Salary Account?", [0.0, 1.0], index=1)
current_account = st.selectbox("Current Account", [0, 1], index=0)
fixed_deposit = st.selectbox("Fixed Deposit", [0, 1], index=1)
recurring_deposit = st.selectbox("Recurring Deposit", [0, 1], index=1)
savings_account = st.selectbox("Savings Account", [0, 1], index=1)

# Process Customer Data
def process_customer_data(json_data, scaler):
    customer_data = pd.DataFrame([json_data])
    clustering_data = customer_data.drop(columns=['customer_id', 'name', 'occupation','gender','marital_status'])
    clustering_data.fillna(0, inplace=True)
    scaled_data = scaler.transform(clustering_data)
    return customer_data, scaled_data

# Predict Customer Segment
def predict_customer_segment(scaled_data, kmeans):
    return kmeans.predict(scaled_data)[0]

# Recommend Product
def recommend_product_and_loan(json_data, kmeans, scaler, multi_target_classifier):
    customer_data, scaled_data = process_customer_data(json_data, scaler)
    customer_segment = predict_customer_segment(scaled_data, kmeans)
    customer_data['customer_segment'] = customer_segment
    X_classification_prod = customer_data.drop(columns=['customer_id', 'name', 'occupation', 'gender', 'marital_status'])
    prob_credit_card = [estimator.predict_proba(X_classification_prod)[:, 1] for estimator in multi_target_classifier.estimators_]
    product_probabilities = pd.Series({
        'Current Account': prob_credit_card[0][0],
        'Fixed Deposit': prob_credit_card[1][0],
        'Recurring Deposit': prob_credit_card[2][0],
        'Savings Account': prob_credit_card[3][0]
    })
    return product_probabilities.idxmax(),customer_segment

if st.button("Recommend Product"):
    user_data = {
        'customer_id': customer_id, 'name': name, 'age': age, 'gender': gender, 'marital_status': marital_status, 'occupation': occupation,
        'max_dpd': max_dpd, 'default_status': default_status, 'transaction_amount': transaction_amount, 'account_balance': account_balance, 'is_salary': is_salary,
        'Current Account': current_account, 'Fixed Deposit': fixed_deposit, 'Recurring Deposit': recurring_deposit, 'Savings Account': savings_account
    }
    recommendation,customer_segment = recommend_product_and_loan(user_data, kmeans, scaler, multi_target_classifier)
    st.success(f"Recommended Product: {recommendation} ")
    st.success(f"Customer segment is : {customer_segment}")
