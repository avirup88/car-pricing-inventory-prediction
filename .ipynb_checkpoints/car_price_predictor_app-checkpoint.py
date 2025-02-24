import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cache model loading
@st.cache_resource
def load_models():
    return {
        "price_model": joblib.load('car_pricing_model.pkl'),
        "days_model": joblib.load('car_days_model.pkl'),
        "label_encoders": joblib.load('label_encoders.pkl'),
        "scaler": joblib.load('scaler.pkl')
    }

# Load models once
models = load_models()
price_model = models["price_model"]
days_model = models["days_model"]
label_encoders = models["label_encoders"]
scaler = models["scaler"]

# Available options
makes = {
    "Toyota": ["Corolla", "Camry", "RAV4", "Highlander", "Prius"],
    "Honda": ["Civic", "Accord", "CR-V", "Pilot", "Fit"],
    "Ford": ["Mustang", "Focus", "Escape", "F-150", "Explorer"],
    "BMW": ["X5", "X3", "3 Series", "5 Series", "7 Series"],
    "Mercedes-Benz": ["C-Class", "E-Class", "GLC", "GLE", "S-Class"],
    "Audi": ["A3", "A4", "A6", "Q5", "Q7"],
    "Tesla": ["Model 3", "Model S", "Model X", "Model Y"],
    "Nissan": ["Altima", "Sentra", "Rogue", "Pathfinder", "370Z"],
    "Hyundai": ["Elantra", "Sonata", "Tucson", "Santa Fe", "Kona"],
    "Volkswagen": ["Jetta", "Passat", "Tiguan", "Atlas", "Golf"]
}

fuel_types = ["Petrol", "Diesel", "Electric", "Hybrid"]
transmissions = ["Manual", "Automatic", "CVT"]

# Function to find make from model
def get_make_from_model(selected_model):
    for make, models in makes.items():
        if selected_model in models:
            return make
    return None

# Initialize session state
if "selected_make" not in st.session_state:
    st.session_state.selected_make = "Toyota"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = makes["Toyota"][0]

# Preprocessing function
def preprocess_input(make, model, fuel_type, transmission, mileage, year, engine_size):
    encoded_features = {
        "make": list(makes.keys()).index(make),
        "model": makes[make].index(model),
        "fuel_type": fuel_types.index(fuel_type),
        "transmission": transmissions.index(transmission)
    }
    
    numerical_data = np.array([[mileage, year, engine_size]])
    numerical_data_df = pd.DataFrame(numerical_data, columns=["mileage", "year", "engine_size"])
    numerical_data_scaled = scaler.transform(numerical_data_df)
    
    features = np.array([[
        encoded_features['make'],
        encoded_features['model'],
        encoded_features['fuel_type'],
        encoded_features['transmission'],
        *numerical_data_scaled[0]
    ]])
    return features

# Prediction function
def predict_price_and_days(features):
    base_price = price_model.predict(features)[0]
    base_days = days_model.predict(features)[0]

    noise_level = 0.01
    features_variations = np.tile(features, (50, 1)) + np.random.normal(0, noise_level, (50, features.shape[1]))
    
    price_predictions = price_model.predict(features_variations)
    days_predictions = days_model.predict(features_variations)
    
    price_confidence = max(0, min(100, 100 - (np.std(price_predictions) / (base_price + 1e-6)) * 100))
    days_confidence = max(0, min(100, 100 - (np.std(days_predictions) / (base_days + 1e-6)) * 100))
    
    return base_price, base_days, price_confidence, days_confidence

# UI Layout
def main():
    st.markdown("<h2 style='text-align: center;'>🚗 Car Price and Sales Duration Predictor</h2>", unsafe_allow_html=True)
    st.markdown("#### Enter the car details below:")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # Dropdown for Make
        selected_make = st.selectbox(
            "Car Make", 
            list(makes.keys()), 
            index=list(makes.keys()).index(st.session_state.selected_make), 
            key="selected_make"
        )

        # If Make changes, update the available models
        if selected_make != st.session_state.selected_make:
            st.session_state.selected_make = selected_make
            st.session_state.selected_model = makes[selected_make][0]  # Reset model to the first option
        
        # Dropdown for Model (dependent on Make)
        selected_model = st.selectbox(
            "Car Model", 
            makes[selected_make], 
            index=makes[selected_make].index(st.session_state.selected_model) if st.session_state.selected_model in makes[selected_make] else 0, 
            key="selected_model"
        )

        # If Model changes, ensure the correct Make is set
        correct_make = get_make_from_model(selected_model)
        if correct_make and correct_make != selected_make:
            st.session_state.selected_make = correct_make
        
        selected_fuel_type = st.selectbox("Fuel Type", fuel_types, key="selected_fuel_type")
        selected_transmission = st.selectbox("Transmission", transmissions, key="selected_transmission")

    with col2:
        mileage = st.slider("Mileage (km)", 1000, 500000, key="mileage")
        year = st.slider("Year of Manufacture", 2005, 2025, key="year")
        engine_size = st.slider("Engine Size (L)", 0.8, 8.0, key="engine_size")
        acquisition_cost = st.slider("Acquisition Cost (€)", 5000.0, 100000.0, key="acquisition_cost", step=1000.0)

    st.divider()

    col3, col4 = st.columns([1, 1])
    predict_button = col3.button("📊 Predict Price & Sales Duration")
    reset_button = col4.button("🔄 Reset Values")

    if reset_button:
        st.rerun()

    if predict_button:
        features = preprocess_input(
            st.session_state.selected_make, 
            st.session_state.selected_model, 
            selected_fuel_type, 
            selected_transmission, 
            mileage, year, engine_size
        )
        predicted_price, predicted_days, price_confidence, days_confidence = predict_price_and_days(features)
        profit = predicted_price - acquisition_cost
        
        st.success(f"#### 💰 Estimated Sale Price: **€{predicted_price:,.2f}** (Confidence: {price_confidence:.2f}%)")
        st.info(f"#### ⏳ Estimated Days in Inventory: **{predicted_days:.0f} days** (Confidence: {days_confidence:.2f}%)")
        
        if profit >= 0:
            st.success(f"#### 📈 Expected Profit: **€{profit:,.2f}**")
        else:
            st.error(f"#### 📉 Expected Loss: **€{profit:,.2f}**")

if __name__ == "__main__":
    main()
