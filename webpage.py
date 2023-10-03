import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Load the pre-trained XGBoost model
regressor = XGBRegressor()
regressor.load_model("xgboost_model.json")  # You should save your trained model to a file

# Define a CSS style for the result box
result_style = """
    background-color: #f0f0f0;
    padding: 5px;
    border: 2px solid #333;
    border-radius: 5px;
    font-size: 16px;  /* Increase the font size for the result box */
    display: inline-block;
"""

# Center-align the title
st.markdown("<h1 style='text-align: center;'># Car Selling Price Prediction</h1>", unsafe_allow_html=True)

# Add a small project description
st.write("This project is a car selling price prediction tool that uses machine learning to estimate the selling price of a used car based on various input features.")

# User input for Year of Manufacture
year = st.slider("Year of Manufacture", 1950, 2023, 2023)

# User input for Present Price (in lakh INR)
present_price = st.number_input("Present Price (in lakh INR)")

# User input for Kilometers Driven
kms_driven = st.number_input("Kilometers Driven")

# Manually specify the available options for Fuel Type, Seller Type, and Transmission Type
fuel_type_options = ['CNG', 'Diesel', 'Petrol']
seller_type_options = ['Dealer', 'Individual']
transmission_options = ['Automatic', 'Manual']

# User input for Fuel Type
fuel_type = st.selectbox("Fuel Type", fuel_type_options)

# User input for Seller Type
seller_type = st.selectbox("Seller Type", seller_type_options)

# User input for Transmission Type
transmission = st.selectbox("Transmission Type", transmission_options)

# Slider for Number of Owners (limited to 0 to 5)
number_of_owners = st.slider("Number of Owners", 0, 5, 0)

# Fit LabelEncoders to the predefined options
le_fuel = LabelEncoder()
le_fuel.fit(fuel_type_options)

le_seller_type = LabelEncoder()
le_seller_type.fit(seller_type_options)

le_transmission = LabelEncoder()
le_transmission.fit(transmission_options)

# Predict button to trigger prediction
if st.button("Predict"):
    # Transform user input
    fuel_type_encoded = le_fuel.transform([fuel_type])[0]
    seller_type_encoded = le_seller_type.transform([seller_type])[0]
    transmission_encoded = le_transmission.transform([transmission])[0]

    # Prepare input data for prediction
    input_data = np.array([year, present_price, kms_driven, fuel_type_encoded, seller_type_encoded, transmission_encoded, number_of_owners]).reshape(1, -1)

    # Make prediction
    predicted_price = regressor.predict(input_data)[0]

    # Display prediction and formatted result box in the same line with increased font size
    st.markdown(
        f'<div style="text-align: center;"><span style="font-weight: bold; font-size: 24px;">Predicted Selling Price: ₹ {predicted_price:.2f} lakhs</span></div>'
        f'<div style="{result_style}"></div>',
        unsafe_allow_html=True
    )
