import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib

def generate_synthetic_data(num_samples=1_000_000, sold_output="./datasets/sold_cars.csv", unsold_output="./datasets/unsold_cars.csv"):
    """
    Generates a synthetic dataset for car sales, including attributes like brand, model, year, mileage, condition, and pricing.
    Saves the dataset as separate files for sold and unsold cars.
    
    Parameters:
    num_samples (int): Number of synthetic data samples to generate.
    sold_output (str): Path to save the sold cars dataset.
    unsold_output (str): Path to save the unsold cars dataset.
    
    Returns:
    tuple: DataFrames for sold and unsold cars.
    """
    print(f"Generating synthetic dataset with {num_samples} samples...")
    
    car_brands = {
        "Toyota": ["Corolla", "Camry", "RAV4"],
        "Honda": ["Civic", "Accord", "CR-V"],
        "Ford": ["Focus", "Fusion", "Escape"],
        "BMW": ["3 Series", "5 Series", "X5", "7 Series"],
        "Mercedes": ["C-Class", "E-Class", "GLC", "S-Class"],
        "Tesla": ["Model 3", "Model S", "Model X", "Model Y"],
        "Porsche": ["911", "Cayenne", "Panamera", "Taycan"],
        "Lexus": ["ES", "RX", "LS", "GX"]
    }
    
    np.random.seed(42)
    years = list(range(2010, 2024))
    year_probabilities = [0.04] * 7 + [0.06, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16]
    
    data = []
    for _ in range(num_samples):
        brand = random.choice(list(car_brands.keys()))
        model = random.choice(car_brands[brand])
        year = np.random.choice(years, p=year_probabilities)
        mileage = max(5_000, min(200_000, int(np.random.normal(60_000, 30_000))))
        condition = np.random.choice(["Excellent", "Good", "Fair", "Poor"], p=[0.2, 0.5, 0.2, 0.1])
        purchase_cost = int(np.random.normal(50_000, 15_000)) if brand in ["BMW", "Mercedes", "Tesla", "Porsche", "Lexus"] else int(np.random.normal(25_000, 10_000))
        purchase_cost = max(5_000, purchase_cost)
        competitor_price = purchase_cost * np.random.uniform(1.05, 1.25)
        days_on_market = min(int(np.random.exponential(scale=60)), 365)
        demand_score = max(1, min(100, int(np.random.normal(50, 20))))
        current_price = purchase_cost * np.random.uniform(1.2, 1.5) if brand in ["BMW", "Mercedes", "Tesla", "Porsche", "Lexus"] else purchase_cost * np.random.uniform(1.1, 1.4)
        sold_price = current_price * np.random.uniform(0.85, 0.98) if np.random.rand() < 0.75 else np.nan
        discount_given = (current_price - sold_price) if not np.isnan(sold_price) else 0
        
        data.append([brand, model, year, mileage, condition, purchase_cost, competitor_price, days_on_market, demand_score, current_price, sold_price, discount_given])
    
    df = pd.DataFrame(data, columns=[
        "Brand", "Model", "Year", "Mileage", "Condition", "Purchase_Cost", "Competitor_Price", 
        "Days_On_Market", "Demand_Score", "Current_Price", "Sold_Price", "Discount_Given"
    ])
    df["Sold"] = df["Sold_Price"].notna().astype(int)

    sold_cars = df.dropna(subset=["Sold_Price"])
    unsold_cars = df[df["Sold_Price"].isna()].copy()
    
    sold_cars.to_csv(sold_output, index=False)
    unsold_cars.to_csv(unsold_output, index=False)
    
    print(f"Sold cars dataset saved at {sold_output}")
    print(f"Unsold cars dataset saved at {unsold_output}")
    
    return sold_cars, unsold_cars

def train_model(sold_cars, model_path="./models/car_price_model.pkl"):
    """
    Trains an XGBoost regression model to predict car selling prices based on sold car data.
    Saves the trained model to the specified path.
    
    Parameters:
    sold_cars (DataFrame): Dataset containing sold cars.
    model_path (str): Path to save the trained model.
    """
    print("Training model with sold cars data...")
    
    features = [
        "Brand", "Model", "Year", "Mileage", "Condition", "Purchase_Cost", "Competitor_Price", 
        "Days_On_Market", "Demand_Score", "Current_Price"
    ]
    
    X = pd.get_dummies(sold_cars[features], columns=["Brand", "Model", "Condition"], drop_first=True)
    y = sold_cars["Sold_Price"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost model...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Model Performance: MAE={mae:.2f}, RMSE={rmse:.2f}")
    
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    sold_cars, unsold_cars = generate_synthetic_data()
    train_model(sold_cars)
