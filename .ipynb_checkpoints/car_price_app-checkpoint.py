import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

# Constants
MODEL_PATH = "./models/car_price_model.pkl"


def format_currency(value):
    """Format a monetary value consistently in Euros."""
    if value >= 1e9:
        return f"{value / 1e9:.2f} B€"
    elif value >= 1e6:
        return f"{value / 1e6:.2f} M€"
    else:
        return f"{value:,.2f} €"


@st.cache_resource()
def load_model():
    """Load the trained machine learning model."""
    return joblib.load(MODEL_PATH)


@st.cache_data()
def load_data(uploaded_file):
    """Load and preprocess the dataset from the uploaded CSV file."""
    df = pd.read_csv(uploaded_file)
    return df


def preprocess_data(unsold_cars, features, model):
    """
    Preprocess the data for prediction by applying one-hot encoding
    and ensuring feature consistency with the trained model.
    """
    X_unsold = pd.get_dummies(
        unsold_cars[features],
        columns=["Brand", "Model", "Condition"],
        drop_first=True,
    )
    # Ensure that the feature set matches the model's expected features
    return X_unsold.reindex(columns=model.get_booster().feature_names, fill_value=0)


def predict_prices(model, X_unsold):
    """Predict the sales prices for unsold cars."""
    return model.predict(X_unsold)


def calculate_profits(unsold_cars):
    """Calculate the expected profit per car."""
    unsold_cars["Expected_Profit"] = unsold_cars["Predicted_Sold_Price"] - unsold_cars["Purchase_Cost"]
    return unsold_cars


def main():
    # Configure Streamlit UI
    st.set_page_config(page_title="Car Sales Price Prediction App", layout="wide")
    st.sidebar.header("Application Controls")
    st.title("🚗 Car Sales Price Prediction & Profit Analysis")
    st.markdown("---")

    # Load model
    model = load_model()

    # Upload dataset
    uploaded_file = st.file_uploader("📂 Upload your dataset (CSV file)", type=["csv"])
    if uploaded_file:
        unsold_cars = load_data(uploaded_file)

        # Define features used for prediction
        features = [
            "Brand", "Model", "Year", "Mileage", "Condition", "Purchase_Cost",
            "Competitor_Price", "Days_On_Market", "Demand_Score", "Current_Price"
        ]
        X_unsold = preprocess_data(unsold_cars, features, model)

        # Perform predictions and calculate profits
        unsold_cars["Predicted_Sold_Price"] = predict_prices(model, X_unsold)
        unsold_cars = calculate_profits(unsold_cars)

        # Aggregate overall dataset results
        total_cars = len(unsold_cars)
        total_predicted_revenue = unsold_cars["Predicted_Sold_Price"].sum()
        total_expected_profit = unsold_cars["Expected_Profit"].sum()
        avg_expected_profit_percentage = (total_expected_profit / unsold_cars["Purchase_Cost"].sum()) * 100
        avg_predicted_price = unsold_cars["Predicted_Sold_Price"].mean()
        avg_profit_per_car = unsold_cars["Expected_Profit"].mean()

        # Display overall results in the sidebar
        st.sidebar.subheader("📊 Overall Aggregate Results")
        st.sidebar.write(f"**Total Unsold Cars:** {total_cars:,}")
        st.sidebar.write(f"**Total Predicted Revenue:** {format_currency(total_predicted_revenue)}")
        st.sidebar.write(f"**Total Expected Profit:** {format_currency(total_expected_profit)}")
        st.sidebar.write(f"**Average Predicted Selling Price:** {format_currency(avg_predicted_price)}")
        st.sidebar.write(f"**Average Profit Per Car:** {format_currency(avg_profit_per_car)}")
        st.sidebar.write(f"**Average Expected Profit Percentage:** {avg_expected_profit_percentage:.1f} %")

        # --- Filtering Options ---
        # Select brand filter
        brands = unsold_cars["Brand"].unique()
        selected_brand = st.sidebar.selectbox("Select Brand", options=["All"] + list(brands))

        # Dynamically select model options based on chosen brand
        if selected_brand != "All":
            models = unsold_cars[unsold_cars["Brand"] == selected_brand]["Model"].unique()
        else:
            models = unsold_cars["Model"].unique()
        selected_model = st.sidebar.selectbox("Select Model", options=["All"] + list(models))

        # Filter the dataset accordingly
        filtered_cars = unsold_cars.copy()
        if selected_brand != "All":
            filtered_cars = filtered_cars[filtered_cars["Brand"] == selected_brand]
        if selected_model != "All":
            filtered_cars = filtered_cars[filtered_cars["Model"] == selected_model]

        # Aggregate filtered dataset results
        filtered_total_cars = len(filtered_cars)
        filtered_total_predicted_revenue = filtered_cars["Predicted_Sold_Price"].sum()
        filtered_total_expected_profit = filtered_cars["Expected_Profit"].sum()
        filtered_avg_predicted_price = filtered_cars["Predicted_Sold_Price"].mean()
        filtered_avg_profit_per_car = filtered_cars["Expected_Profit"].mean()
        filtered_avg_expected_profit_percentage = (
            (filtered_total_expected_profit / filtered_cars["Purchase_Cost"].sum()) * 100
            if filtered_cars["Purchase_Cost"].sum() != 0 else 0
        )

        st.subheader("📊 Filtered Aggregate Results")
        st.write(f"**Total Unsold Cars:** {filtered_total_cars:,}")
        st.write(f"**Total Predicted Revenue:** {format_currency(filtered_total_predicted_revenue)}")
        st.write(f"**Total Expected Profit:** {format_currency(filtered_total_expected_profit)}")
        st.write(f"**Average Predicted Selling Price:** {format_currency(filtered_avg_predicted_price)}")
        st.write(f"**Average Profit Per Car:** {format_currency(filtered_avg_profit_per_car)}")
        st.write(f"**Average Expected Profit Percentage:** {filtered_avg_expected_profit_percentage:.1f} %")

        # --- Detailed Data Table ---
        st.subheader("📋 Detailed Car Data (Sampled)")
        sample_size = min(5000, len(filtered_cars))  # Limit sample to optimize performance
        filtered_cars_sampled = filtered_cars[features + ['Predicted_Sold_Price', 'Expected_Profit']]
        filtered_cars_sampled = filtered_cars_sampled.sample(sample_size, random_state=42).reset_index(drop=True)

        # Consistently format monetary columns in the table
        price_columns = ["Predicted_Sold_Price", "Expected_Profit", "Purchase_Cost", "Competitor_Price", "Current_Price"]
        for col in price_columns:
            if col in filtered_cars_sampled.columns:
                filtered_cars_sampled[col] = filtered_cars_sampled[col].apply(format_currency)
        st.data_editor(filtered_cars_sampled, hide_index=True)

        # --- Visual Business Insights ---
        st.subheader("📊 Business Insights")

        # Distribution: Predicted Sales Price
        st.write("### Distribution of Predicted Sales Price")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_cars["Predicted_Sold_Price"], bins=30, kde=True, color='blue')
        ax.set_xlabel("Predicted Sales Price (€)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        # Distribution: Expected Profit
        st.write("### Distribution of Expected Profit")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_cars["Expected_Profit"], bins=30, kde=True, color='orange')
        ax.set_xlabel("Expected Profit (€)")
        ax.set_ylabel("Count")
        st.pyplot(fig)


if __name__ == "__main__":
    main()
