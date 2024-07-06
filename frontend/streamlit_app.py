import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://localhost:8000"

st.title("AI-Powered Personal Finance Advisor")

# File upload for receipt
uploaded_file = st.file_uploader("Upload a receipt", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{API_URL}/process_receipt", files=files)
    if response.status_code == 200:
        st.success(f"Expense category: {response.json()['category']}")

# Natural language query
query = st.text_input("Ask a financial question")
if query:
    response = requests.post(f"{API_URL}/process_query", json={"query": query})
    if response.status_code == 200:
        st.write(f"Intent: {response.json()['intent']}")

# Budget analysis
st.subheader("Budget Analysis")
income = st.number_input("Monthly Income", min_value=0.0, step=100.0)

if st.button("Analyze Budget"):
    response = requests.get(f"{API_URL}/analyze_budget", params={"income": income})
    if response.status_code == 200:
        analysis = response.json()
        st.write(f"Total Expenses: ${analysis['total_expenses']:.2f}")
        st.write(f"Savings: ${analysis['savings']:.2f}")
        st.write(f"Savings Rate: {analysis['savings_rate']:.2f}%")
        
        df = pd.DataFrame(list(analysis['category_totals'].items()), columns=['Category', 'Amount'])
        fig = px.pie(df, values='Amount', names='Category', title='Expense Breakdown')
        st.plotly_chart(fig)

# Investment recommendation
st.subheader("Investment Recommendation")
risk_tolerance = st.selectbox("Risk Tolerance", ["low", "medium", "high"])
investment_horizon = st.slider("Investment Horizon (years)", 1, 30, 5)

if st.button("Get Recommendation"):
    response = requests.get(f"{API_URL}/investment_recommendation", 
                            params={"risk_tolerance": risk_tolerance, "investment_horizon": investment_horizon})
    if response.status_code == 200:
        st.write(response.json()['recommendation'])

# Model Performance
st.subheader("Model Performance")
if st.button("Check Model Accuracy"):
    response = requests.get(f"{API_URL}/model_accuracy")
    if response.status_code == 200:
        accuracy = response.json()['accuracy']
        st.write(f"Model Accuracy: {accuracy:.2f}")
