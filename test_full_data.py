import numpy as np
import pandas as pd
import os
import joblib
from scipy.stats import randint, uniform
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV, 
    cross_val_score, 
    train_test_split
)
from load_disease_data import load_dataset
from sklearn.linear_model import LogisticRegression


def main() -> None:
    X, y = load_dataset(use_full_data=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.20)

    model = LogisticRegression(random_state=42)

    scores = cross_val_score(
        model, X_train, y_train, cv=4, scoring="accuracy", n_jobs=-1
    )
    mean = scores.mean()
    std = scores.std()
    print(f"Model accuracy on full dataset: {mean:.4f} (+/- {std * 2:.4f})")

    model.fit(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"\nTest set accuracy: {test_accuracy:.4f}")

     # Create a directory to save models
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Save model to directory using joblib
    model_filename = os.path.join(model_dir, "best_logistic_regression.joblib")
    joblib.dump(model, model_filename)


if __name__ == "__main__":
    main()
