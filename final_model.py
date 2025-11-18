import numpy as np
import pandas as pd
import os
import joblib
from scipy.stats import randint, uniform
# Removed duplicate imports
from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV, 
    cross_val_score, 
    train_test_split
)
from exploratory_analysis import load_dataset
from sklearn.linear_model import LogisticRegression


def main() -> None:
    X, y = load_dataset(use_full_data=False)

    # Split into test and training
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.20)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Create logistic regression model (default params)
    baseline_model = LogisticRegression(random_state=42)

    # 4 fold cross validation to evaluate baseline performance
    baseline_scores = cross_val_score(
        baseline_model, X_train, y_train, cv=4, scoring="accuracy", n_jobs=-1
    )
    baseline_mean = baseline_scores.mean()
    baseline_std = baseline_scores.std()
    print(f"Baseline accuracy: {baseline_mean:.4f} (+/- {baseline_std * 2:.4f})")

    # Grid for testing hyperparameters
    # Narrowed down to best params from original parameters after first tests
    grid_param_grid = {
        "C": [1.0, 10.0],
        "solver": ["lbfgs"],
        "max_iter": [50, 100, 150],
    }
    # Create logistic regression model, ensure a fixed random state
    model = LogisticRegression(random_state=42)

    # Perform GridSearch CV
    grid_search = GridSearchCV(
        model,
        grid_param_grid,
        cv=4,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2
    )

    # Fit grid search on the training data
    grid_search.fit(X_train, y_train)
    # Print best parameters, and accuracy, as well as improvement
    print("GridSearchCV completed!")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    print(f"Improvement over baseline: {grid_search.best_score_ - baseline_mean:.4f}")

    # Parameter distributions for RandomizedSearchCV
    # Narrowed down to best params from original parameters after first tests
    random_param = {
        "max_iter": randint(50, 200),
        "C": uniform(0.01, 10),
        "solver": ["lbfgs"],
    }

    # New model fr testing randomized search,
    # ensures both start from same baseline
    model_random = LogisticRegression(random_state=42)

    # Test 40 random parameter combinations
    random_search = RandomizedSearchCV(
        model_random,
        random_param,
        n_iter=20,
        cv=4,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    # Fit the randomized search on training data
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV completed!")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation accuracy: {random_search.best_score_:.4f}")
    print(f"Improvement over baseline: {random_search.best_score_ - baseline_mean:.4f}")

    # Best results from all search methods
    print(f"Baseline accuracy:           {baseline_mean:.4f}")
    print(f"GridSearchCV best accuracy:  {grid_search.best_score_:.4f}")
    print(f"RandomizedSearchCV accuracy: {random_search.best_score_:.4f}")

    # Baseline had almost identical score
    baseline_model.fit(X_train, y_train)
    test_accuracy = baseline_model.score(X_test, y_test)
    print(f"\nTest set accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
