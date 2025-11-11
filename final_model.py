import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV, 
    cross_val_score, 
    train_test_split
)
from exploratory_analysis import feature_engineering, load_dataset
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def transform_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[NDArray, NDArray]:
    
    # Feature engineer both train and test sets
    X_train_engineered = feature_engineering(X_train.copy())
    X_test_engineeered = feature_engineering(X_test.copy())
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_engineered)
    X_test_scaled = scaler.transform(X_test_engineered)
    return X_train_scaled, X_test_scaled

def main() -> None:
    X, y = load_dataset()
    
    # Split into test and training
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.20)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Apply feature engineering and scale features
    X_train_scaled, X_test_scaled = transform_data(X_train, X_test)
    
    # Create logistic regression model (default params)
    baseline_model = LogisticRegression(random_state=42)
    
    # 4 fold cross validation to evaluate baseline performance
    baseline_scores = cross_val_score(
        baseline_model, X_train_scaled, y_train, cv=4, scoring="accuracy", n_jobs=-1
    )
    baseline_mean = baseline_scores.mean()
    baseline_std = baseline_scores.std()
    print(f"Baseline accuracy: {baseline_mean:.4f} (+/- {baseline_std * 2:.4f})")
    
    # Grid for testing hyperparameters
    grid_param_grid = {
        "C": [0.1, 1.0, 10.0],
        "solver": ["lbfgs"],
        "max_iter": [100, 200],
    }
    # Create logistic regression model, ensure fixed random state
    model = LogisticRegression(random_state=42)
    
    # Perform GridSearch CV
    grid_search = GridSearchCV(
        model,
        grid_param_grid,
        cv=4,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search on the training data
    print("Starting GridSearchCV...")
    grid_search.fit(X_train_scaled, y_train)
    # Print best parameters, and accuracy, as well as improvement
    print("GridSearchCV completed!")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    print(f"Improvement over baseline: {grid_search.best_score_ - baseline_mean:.4f}")
    
    # Parameter distributions for RandomizedSearchCV
    random_param = {
        
    }
    
    
    
    
    
    
    