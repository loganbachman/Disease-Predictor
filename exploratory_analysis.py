import numpy as np
import pandas as pd
import os
from sklearn.model_selection import cross_val_score, train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def load_dataset(data_path='data/raw', use_full_data=True):
    # Create path for data
    os.makedirs(data_path, exist_ok=True)
    csv_name = 'Final_Augmented_dataset_Diseases_and_Symptoms.csv'
    csv_path = os.path.join(data_path, csv_name)
    # if data doesn't exist, call Kaggle Api to load
    if not os.path.exists(csv_path):
        print("Downloading kaggle dataset...")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            'dhivyeshrk/diseases-and-symptoms-dataset',
            path=data_path,
            unzip=True
        )

    dataset = pd.read_csv(csv_path)
    dataset.dropna(inplace=True)
    # X (features) is all columns (symptoms) except first (diseases)
    X = dataset.drop('diseases', axis=1)
    # y (target) is the resulting disease, what we are trying to predict
    y = dataset['diseases']

    # filter out diseases that dont appear at least 10 times
    disease_counts = y.value_counts()
    valid_diseases = disease_counts[disease_counts >= 10].index
    valid_mask = y.isin(valid_diseases)
    X = X[valid_mask]
    y = y[valid_mask]

    # if NOT using full dataset
    if not use_full_data:
        # Take a stratified sample of 50K from dataset
        X, _, y, _ = train_test_split(
            X, y,
            train_size=50000, 
            stratify=y, 
            random_state=42
        )

    return X, y


def main():
    X, y = load_dataset()

    # Split sample into training and testing set (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
     X, y,
     test_size=0.2,
     random_state=42
    )


# These are unused so I have them commented out for now

    # Potential best hyperparameter combinations
    # params = {
    #  "n_estimators": [100, 200],
    # "max_depth": [None, 20],
    # "max_features": ["sqrt", None],
    # "max_leaf_nodes": [None, 8]
    # }
    
    # Logistic Regression model
    model2 = LogisticRegression()
    
    # Potential best hyperparameter combinations
    params2 = {
        "C": [0.1, 1.0, 10.0],
        "solver": ["lbfgs", "newton-cg"],
        "max_iter": [100, 200],
    }

    # Do the same but for logistic regression without parameter tuning
    # ~85-86% accuracy (better than random forest)
    base_scores2 = cross_val_score(
     model2, X_train, y_train,
     cv=4, scoring="accuracy"
    )

    print("Cross validation score:", base_scores2)
    print(f"Mean validation score: {np.mean(base_scores2):.4f}")

    # Use GridSearchCV to find best hyperparameter combinations
    # for chosen model (Logistic Regression)
    grid = GridSearchCV(
     model2, params2,
     cv=4, scoring="accuracy",
     n_jobs=-1, verbose=2
    )
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)
    print(f"Tuned mean accuracy: {grid.best_score_:.3f}")
    # Best parameters: {'C': 1.0, 'max_iter': 100, 'solver': 'lbfgs'}
    # The tuned mean accuracy: ~86%


if __name__ == "__main__":
    main()
