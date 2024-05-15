import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load your trained model
def load_model():
    with open('RFClassifier.sav', 'rb') as file:
        data = pickle.load(file)
    return data

model = load_model()

# Set page configuration
st.set_page_config(page_title="Personal Loan Predictor", page_icon=":moneybag:", layout="wide")

# Header Section
# st.image("https://source.unsplash.com/600x200/?money,finance", use_column_width=True)
st.title("Personal Loan Campaign Prediction Tool")
st.markdown("""
Welcome to the **Personal Loan Campaign Prediction Tool**. This application helps banks and financial institutions
determine the likelihood of customers accepting personal loan offers. Simply input the customer details below
and get instant predictions.
""")
st.markdown("---")

# Main Section
col1, col2 = st.columns(2, gap='large')

with col1:
    st.header("Information Panel")
    tabs1, tabs2, tabs3, tabs4 = st.tabs(['Overview', 'Feature', 'Benefits', 'Ideal For'])
    tabs1.subheader("Overview")
    tabs1.markdown("""
                The Personal Loan Prediction App is a cutting-edge tool designed to assist banks, 
                financial advisors, and individual consumers in understanding the likelihood of a customer 
                accepting a personal loan offer. This application uses sophisticated machine learning algorithms 
                to analyze customer data and predict their propensity to take a loan, helping financial institutions 
                tailor their offers more effectively.
    """)

    tabs2.subheader('Features')
    tabs2.markdown("""
    - **Predictive Analytics:** Utilizes historical data and predictive modeling to forecast the likelihood of a customer accepting a personal loan. The model considers various factors, including income level, employment status, credit history, and more.
    - **User-Friendly Interface:** Features an intuitive and easy-to-navigate interface, allowing users to input customer data effortlessly and receive predictions in real-time.
    - **Data-Driven Insights:** Provides insights into the key factors influencing loan acceptance, enabling users to understand what drives customer decisions and how to improve their loan offerings.
    - **Secure and Reliable:** Ensures data security with robust encryption standards, protecting sensitive customer information while processing predictions.
    - **Customizable Scenarios:** Users can modify input parameters to simulate different scenarios and see how changes in customer data might affect their likelihood of accepting a loan.
""")
    
    tabs3.subheader('Benefits')
    tabs3.markdown("""
    - **Enhanced Decision Making:** By knowing the likelihood of loan acceptance, banks and financial advisors can make informed decisions on whom to target, optimizing their marketing strategies and resource allocation.
    - **Increased Conversion Rates:** Targeting the right customers increases the chances of loan acceptance, thereby boosting conversion rates and enhancing overall profitability.
    - **Cost Efficiency:** Reduces the costs associated with broad, non-targeted marketing campaigns by focusing efforts on prospects more likely to accept loans.
    - **Improved Customer Experience:** By understanding customer needs and likelihood to accept offers, institutions can personalize their interactions, enhancing customer satisfaction and loyalty.
""")
    
    tabs4.subheader('Ideal For')
    tabs4.markdown("""
    - Banks and Financial Institutions: Looking to optimize their personal loan offerings and increase the effectiveness of their marketing campaigns.
    - Financial Advisors: Seeking to provide tailored advice to clients considering personal loans.
    - Data Analysts and Marketers: Who require advanced tools to analyze customer behavior and predict market trends.
""")

with col2:
    header = st.empty()
    header.header("Prediction")
    placeholder = st.empty()
    
    # Name Container
    name_container = placeholder.container(border=False)
    name = name_container.text_input(label="name",placeholder="Enter customer's name", label_visibility='hidden')

    # Feature Input
    feature_col1, feature_col2 = name_container.columns(2)
    
    age = feature_col1.number_input("Age", min_value=18, max_value=100, value=30)
    income = feature_col1.number_input("Annual Income ($ thousands)", min_value=10, max_value=1000, value=50)
    family = feature_col1.number_input("Family Size", min_value=1, max_value=10, value=3)
    experience = feature_col1.number_input("Profesional Experience", min_value=0, max_value=20, value=20)
    cda = feature_col1.selectbox("Have Certificate Deposit Account", options=['Yes', 'No'])
    security = feature_col1.selectbox("Have Security Account", options=['Yes', 'No'])


    education = feature_col2.selectbox("Education Level", ("Undergraduate", "Graduate", "Advanced/Professional"))
    mortgage = feature_col2.number_input("Mortgage Value of house ($ thousands)", min_value=0, max_value=1000, value=0)
    credit_card_spending = feature_col2.number_input("Monthly Credit Card Spending ($ thousands)", min_value=0, max_value=100, value=2)
    ccd = feature_col2.selectbox("Have Credit Card Account", options=['Yes', 'No'])
    online = feature_col2.selectbox("Using internet Banking", options=['Yes', 'No'])

    # Convert categorical data to numerical under the hood
    education_mapping = {"Undergraduate": 1, "Graduate": 2, "Advanced/Professional": 3}
    education = education_mapping[education]

    bool_mapping = {"Yes":1, "No":0}
    cda = bool_mapping[cda]
    security = bool_mapping[security]
    ccd = bool_mapping[ccd]
    online = bool_mapping[online]

    btn_placeholder = st.empty()
    btn_predict = btn_placeholder.button("Predict !")



# Prediction section
if btn_predict:
    
    # Inferential Process
    input_data = np.array([[age, income, family, education, mortgage, credit_card_spending, ccd, online, experience, security, cda]])
    input_data = pd.DataFrame(input_data, columns=['Age', 'Income', 'Family', 'Education', 'Mortgage', 'CCAvg', 'CreditCard', 'Online', 'Experience', 'Securities Account','CD Account'])
    prediction = model.predict_proba(input_data)
    prediction = round(prediction[0][1] * 100, 2)

    placeholder.write(f"The customer has {prediction}% to accept the personal loan offer")
    
    if btn_placeholder.button("Predict Again !"):
        st.rerun()


# Footer
st.markdown("---")
st.markdown("""
This tool is designed for illustrative and professional use by financial advisors and should not be considered as personal financial advice.
Contact a financial expert before making significant financial decisions.
""")