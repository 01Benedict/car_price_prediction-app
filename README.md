# Car Price Prediction

## Overview
This project implements a machine learning regression model to predict car prices based on various vehicle features. The model is built and trained using Python with scikit-learn, pandas, and other data science libraries.

## Task
Build a regression model that accurately predicts the price of a car using historical vehicle data and features.

## Dataset
- **File**: `car_price_prediction.csv`
- **Records**: Multiple car entries with various attributes
- **Target Variable**: `price` (the value we want to predict)

## Features Used
The model uses the following features for prediction:
- `production_year` - Year the car was manufactured
- `levy` - Tax/levy amount on the vehicle
- `mileage` - Total kilometers driven
- `cylinders` - Number of cylinders in the engine
- `airbags` - Number of airbags in the vehicle

## Model Details
- **Algorithm**: Linear Regression
- **Library**: scikit-learn
- **Training-Test Split**: 75-25 split (random_state=42)

## Data Processing
### Cleaning Steps:
1. **Column Renaming**: Standardized column names for consistency
   - Example: 'Prod. year' → 'production_year'
   
2. **Data Type Conversions**:
   - Removed hyphens from levy column (replaced '-' with '0')
   - Removed 'km' suffix from mileage column
   - Removed 'Turbo' designation from engine volume column
   - Converted levy and engine_volume to float type
   - Converted mileage to integer type

## Model Performance
The model is evaluated using standard regression metrics:
- **R² Score**: Coefficient of determination (0-1 scale, higher is better)
- **MSE**: Mean Squared Error (average squared prediction errors)
- **RMSE**: Root Mean Squared Error (same units as target variable)

## Visualizations
The notebook includes scatter plots showing relationships between:
- Price vs Levy
- Price vs Production Year
- Price vs Mileage
- Price vs Cylinders
- Price vs Airbags (with regression line)

## Required Libraries
- pandas
- matplotlib
- seaborn
- scikit-learn
- numpy

## Installation

### Using pip
Install required dependencies:
```bash
pip install -r requirements.txt
```

### Using pipenv
Install dependencies using Pipenv:
```bash
pipenv install
```

To activate the virtual environment:
```bash
pipenv shell
```

## Usage
Run the Jupyter notebook to execute the complete pipeline:
```bash
jupyter notebook benedict_car_prediction.ipynb
```

The notebook will:
1. Load and inspect the data
2. Clean and prepare the data
3. Visualize relationships between features and price
4. Train the linear regression model
5. Generate predictions and evaluate performance

## File Structure
```
2526500463_car_prediction/
├── benedict_car_prediction.ipynb      # Main notebook
├── car_price_prediction.csv           # Dataset
├── requirements.txt                   # Dependencies
├── Pipfile                           # Pipenv configuration
└── README.md                         # This file
```

## Notes
- The model assumes a linear relationship between features and car price
- Consider feature scaling or normalization for improved model performance
- Cross-validation could be implemented for more robust evaluation
- Additional preprocessing or feature engineering may improve predictions
