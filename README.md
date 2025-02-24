# Car Price Prediction

## Overview
This repository contains a machine learning project for predicting car sales prices based on various factors. It includes a model training script and a Streamlit-based web application for interactive predictions and profit analysis.

## Features
- **Synthetic Data Generation:** Generates a dataset of sold and unsold cars with relevant attributes.
- **Machine Learning Model:** Trains an XGBoost regression model to predict car prices.
- **Streamlit Web App:** Provides an interactive UI to upload datasets, predict car prices, and visualize insights.
- **Profit Analysis:** Estimates expected revenue and profits for unsold cars.

## Project Structure
```
├── datasets/               # Directory for storing generated datasets
├── models/                 # Directory for saving trained models
├── car_price_prediction_model_training.py  # Model training script
├── car_price_prediction_app.py             # Streamlit web application
├── README.md               # Project documentation
```

## Installation
### Prerequisites
Ensure you have Python installed (>=3.7). Install the required dependencies using:
```bash
pip install -r requirements.txt
```

### Required Libraries
The main dependencies are:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `streamlit`
- `joblib`
- `matplotlib`
- `seaborn`

## Usage
### 1. Generate Synthetic Data
Run the following command to generate synthetic datasets:
```bash
python car_price_prediction_model_training.py
```
This will create `sold_cars.csv` and `unsold_cars.csv` in the `datasets/` directory.

### 2. Train the Model
Once the dataset is generated, train the XGBoost model using:
```bash
python car_price_prediction_model_training.py
```
The trained model will be saved in the `models/` directory.

### 3. Run the Web Application
Start the Streamlit web application by running:
```bash
streamlit run car_price_prediction_app.py
```
Upload a dataset and explore predictions and insights interactively.

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

## Authors
Developed by Avirup Chakraborty

