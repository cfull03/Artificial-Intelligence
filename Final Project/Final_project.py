
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import random
import os
from datetime import datetime

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame):
    target = 'PastProfit'
    features = df.drop(columns=[target])

    categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = features.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    X = features
    y = df[target]

    return X, y, preprocessor

def train_models(X, y, preprocessor, performance_file):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'LinearRegression': Pipeline([
            ('preprocessor', preprocessor),
            ('pca', PCA(n_components=0.95)),
            ('regressor', LinearRegression())
        ]),
        'RandomForest': Pipeline([
            ('preprocessor', preprocessor),
            ('pca', PCA(n_components=0.95)),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
    }

    trained_models = {}

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    performance_file.write(f"\n==== Model Performance at {timestamp} ====\n")

    print("\nModel Performance Summary:")
    print("==========================")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        summary = (f"Model: {name}\n"
                   f"  - Mean Squared Error (MSE): {mse:.2f}\n"
                   f"  - R-squared (R2 Score): {r2:.2f}\n")
        print(summary)
        performance_file.write(summary + '\n')
        trained_models[name] = model

    return trained_models

def predict_best_location_holiday(df: pd.DataFrame, model_pipeline, results_file):
    unique_locations = df['Location'].unique()
    unique_holidays = df['Holiday'].unique()

    base_sample = df.sample(1, random_state=42).drop(columns=['Location', 'Holiday', 'PastProfit'])

    prediction_records = []

    for loc in unique_locations:
        for hol in unique_holidays:
            sample = base_sample.copy()
            sample['Location'] = loc
            sample['Holiday'] = hol

            pred_profit = model_pipeline.predict(sample)[0]

            prediction_records.append({
                'Location': loc,
                'Holiday': hol,
                'PredictedProfit': pred_profit
            })

    predictions_df = pd.DataFrame(prediction_records)
    best_combo = predictions_df.sort_values(by='PredictedProfit', ascending=False).head(1)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_summary = f"\nBest Location and Holiday Prediction ({timestamp}):\n" + best_combo.to_string(index=False) + '\n'

    print(result_summary)
    results_file.write(result_summary)

    return predictions_df

if __name__ == "__main__":
    df = pd.read_csv('generated_profit_data.csv')

    with open('performance_summary.txt', 'a') as performance_file, open('results_summary.txt', 'a') as results_file:
        X, y, preprocessor = preprocess_data(df)
        models = train_models(X, y, preprocessor, performance_file)

        best_model = models['RandomForest']
        predict_best_location_holiday(df, best_model, results_file)
