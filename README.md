# Task 1 - House Price Prediction

## Objective
Implement a Linear Regression model to predict house prices using:
- Square footage
- Number of bedrooms
- Number of bathrooms

## Dataset
Kaggle House Prices - Advanced Regression Techniques.

Download `train.csv` and place it in the same folder as `task1_linear_regression.py`.

## How to run
Install:
```bash
pip install pandas matplotlib scikit-learn
```

Then run:
```bash
python task1_linear_regression.py
```

## Model
A Multiple Linear Regression model is used with:
- `GrLivArea` → living area in square feet
- `BedroomAbvGr` → number of bedrooms
- `Bathrooms` → calculated from full and half bathrooms

The model is evaluated using MAE, MSE and R² score.
