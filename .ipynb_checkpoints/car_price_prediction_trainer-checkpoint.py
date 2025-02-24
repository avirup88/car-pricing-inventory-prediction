import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from faker import Faker

# Initialize Faker
fake = Faker()
Faker.seed(42)

# Car brands and models with price multipliers, depreciation rates, demand factors, resale scores, and competition factors
car_brands_models = {
    'Toyota': (['Corolla', 'Camry', 'RAV4', 'Highlander', 'Prius'], 1.0, 0.85, 1.2, 1.3, 1.1),
    'Honda': (['Civic', 'Accord', 'CR-V', 'Pilot', 'Fit'], 1.0, 0.84, 1.1, 1.2, 1.05),
    'Ford': (['Mustang', 'Focus', 'Escape', 'F-150', 'Explorer'], 1.1, 0.83, 1.0, 1.0, 1.0),
    'BMW': (['X5', 'X3', '3 Series', '5 Series', '7 Series'], 1.5, 0.80, 0.8, 0.8, 0.9),
    'Tesla': (['Model 3', 'Model S', 'Model X', 'Model Y'], 1.6, 0.88, 1.5, 1.5, 1.3),
    'Mercedes-Benz': (['C-Class', 'E-Class', 'GLC', 'GLE', 'S-Class'], 1.5, 0.79, 0.9, 0.9, 0.95),
    'Audi': (['A3', 'A4', 'A6', 'Q5', 'Q7'], 1.4, 0.81, 0.95, 0.85, 0.92),
    'Chevrolet': (['Malibu', 'Camaro', 'Silverado', 'Tahoe', 'Equinox'], 1.1, 0.82, 1.0, 1.0, 1.02),
    'Nissan': (['Altima', 'Sentra', 'Rogue', 'Pathfinder', '370Z'], 1.0, 0.83, 1.05, 1.05, 1.08),
    'Hyundai': (['Elantra', 'Sonata', 'Tucson', 'Santa Fe', 'Kona'], 0.9, 0.86, 1.3, 1.1, 1.15),
    'Kia': (['Forte', 'Optima', 'Sportage', 'Sorento', 'Telluride'], 0.9, 0.87, 1.25, 1.1, 1.12),
    'Volkswagen': (['Jetta', 'Passat', 'Tiguan', 'Atlas', 'Golf'], 1.2, 0.84, 1.1, 1.05, 1.07)
}


def generate_synthetic_data(num_samples=100000, save_path="synthetic_car_sales_realistic.csv"):
    print(f"Generating synthetic dataset with {num_samples} samples...")

    makes = list(car_brands_models.keys())
    car_make = np.random.choice(makes, num_samples)
    car_model = []
    price_multipliers = []
    depreciation_rates = []
    demand_factors = []
    resale_scores = []
    competition_factors = []
    
    for make in car_make:
        models, multiplier, depreciation, demand, resale, competition = car_brands_models[make]
        car_model.append(np.random.choice(models))
        price_multipliers.append(multiplier)
        depreciation_rates.append(depreciation)
        demand_factors.append(demand)
        resale_scores.append(resale)
        competition_factors.append(competition)

    data = {
        'make': car_make,
        'model': car_model,
        'year': np.random.randint(2005, 2023, num_samples),
        'mileage': np.random.randint(5000, 200000, num_samples),
        'fuel_type': np.random.choice(['Petrol', 'Diesel', 'Electric', 'Hybrid'], num_samples, p=[0.5, 0.3, 0.1, 0.1]),
        'transmission': np.random.choice(['Automatic', 'Manual', 'CVT'], num_samples, p=[0.6, 0.3, 0.1]),
        'engine_size': np.random.uniform(1.0, 5.0, num_samples).round(1),
        'seller_name': [fake.name() for _ in range(num_samples)],
        'location': [fake.city() for _ in range(num_samples)],
        'listing_date': [fake.date_between(start_date='-2y', end_date='today') for _ in range(num_samples)]
    }

    df = pd.DataFrame(data)
    df['car_age'] = 2025 - df['year']
    df['acquisition_cost'] = (
        np.random.uniform(5000, 50000, num_samples) * np.array(price_multipliers) * (np.array(depreciation_rates) ** df['car_age'])
    )
    df['sale_price'] = df['acquisition_cost'] * np.random.uniform(0.85, 1.2, num_samples)
    df['listing_date'] = pd.to_datetime(df['listing_date'])
    df['seasonality_factor'] = np.where(df['listing_date'].dt.month.isin([6, 7, 8]), 1.2, 1.0)
    df['mileage_penalty'] = np.where(df['mileage'] > 150000, 1.3, np.where(df['mileage'] > 100000, 1.15, 1.0))
    df['market_price_factor'] = np.where(df['sale_price'] < df['acquisition_cost'] * 1.05, 0.8, 1.0)
    df['economic_trend_factor'] = np.random.uniform(0.9, 1.1, num_samples)
    df['days_in_inventory'] = np.clip(
        (np.random.randint(5, 120, num_samples) + df['car_age'] * 2 - (df['sale_price'] / 5000) * np.array(demand_factors)) *
        df['seasonality_factor'] * df['mileage_penalty'] * df['market_price_factor'] * df['economic_trend_factor'] * np.array(resale_scores) * np.array(competition_factors),
        5, 180
    )

    print(df.head(3))

    df.to_csv(save_path, index=False)
    print(f"Synthetic dataset with {num_samples} rows saved at {save_path}")


def train_models(data_path="synthetic_car_sales_realistic.csv"):
    print("Loading dataset...")
    df = pd.read_csv(data_path)

    print(df.head(3))

    # Drop non-numeric columns
    df.drop(columns=['seller_name', 'location', 'listing_date'], inplace=True)

    categorical_features = ['make', 'model', 'fuel_type', 'transmission']
    numerical_features = ['mileage', 'year', 'engine_size']
    target_price, target_days = 'sale_price', 'days_in_inventory'

    print("Splitting dataset...")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    print("Encoding categorical features...")
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        df_test[col] = df_test[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        label_encoders[col] = le

    print("Scaling numerical features...")
    scaler = MinMaxScaler()
    df_train[numerical_features] = scaler.fit_transform(df_train[numerical_features])
    df_test[numerical_features] = scaler.transform(df_test[numerical_features])

    def train_xgboost(X_train, y_train):
        model = xgb.XGBRegressor(learning_rate = 0.01, n_estimators = 500, max_depth = 6, random_state=42)
        #model = RandomForestRegressor(n_estimators = 200, max_depth = 10, random_state=42)
        model.fit(X_train, y_train)
        return model

    print("Training models...")
    price_model = train_xgboost(df_train[categorical_features + numerical_features], df_train[target_price])
    days_model = train_xgboost(df_train[categorical_features + numerical_features], df_train[target_days])

    def evaluate_model(model, X_test, y_test, name):
        y_pred = model.predict(X_test)
        print(f"\n{name} Model Performance:")
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

    evaluate_model(price_model, df_test[categorical_features + numerical_features], df_test[target_price], "Price Prediction")
    evaluate_model(days_model, df_test[categorical_features + numerical_features], df_test[target_days], "Days in Inventory Prediction")

    print("\nSaving models and preprocessing objects...")
    joblib.dump(price_model, 'car_pricing_model.pkl')
    joblib.dump(days_model, 'car_days_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Models trained and saved successfully.")

if __name__ == "__main__":
    #generate_synthetic_data(num_samples=100000)
    train_models()
