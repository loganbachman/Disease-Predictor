import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

api = KaggleApi()
api.authenticate()

def load_dataset():
    api.dataset_download_files('dhivyeshrk/diseases-and-symptoms-dataset', unzip=True)
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    dataset = pd.read_csv('Final_Augmented_dataset_Diseases_and_Symptoms.csv')
    dataset.dropna(inplace=True)
    
    # X (features) is all columns (symptoms) except first (diseases)
    X = dataset.drop('diseases', axis=1)
    # y (target) is the resulting disease, what we are trying to predict
    y = dataset['diseases']
    
    # filter out diseases that dont appear at least 2 times so we can stratify
    disease_counts = y.value_counts()
    valid_diseases = disease_counts[disease_counts >= 2].index
    valid_mask = y.isin(valid_diseases)
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Take a stratified sample of 25K from dataset
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=50000, stratify=y, random_state=42)
    
    # filter out diseases that dont appear at least 5 times
    disease_counts = y_sample.value_counts()
    valid_diseases = disease_counts[disease_counts >= 10].index
    valid = y_sample.isin(valid_diseases)
    
    # assign X and y to filtered diseases
    X_sample = X_sample[valid]
    y_sample = y_sample[valid]
    return X_sample, y_sample


def main():
    X, y = load_dataset()
    
    # Split sample into training and testing set (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.20)



    # Assign model to random forest
    model = RandomForestClassifier(random_state=42)
    # Potential best hyperparameter combinations
    params = {
        "n_estimators": [100, 200],
        "max_depth": [None, 20],
        "max_features": ["sqrt", None],
        "max_leaf_nodes": [None, 8]
    }
    
    # Logistic Regression model
    model2 = LogisticRegression()
    
    # Potential best hyperparameter combinations
    params2 = {
        "C": [0.1, 1.0, 10.0],
        "solver": ["lbfgs", "newton-cg"],
        "max_iter": [100, 200],
    }
    
    # Cross-fold validation and get our default score for randomforest (without parameter tuning)
    # ~82% accuracy
    #base_scores = cross_val_score(model, X_train, y_train, cv=4, scoring="accuracy")
    #print("Cross validation score:", base_scores)
    #print(f"Mean validation score: {np.mean(base_scores):.4f}")
    
    # Do the same but for logistic regression without parameter tuning
    # ~85-86% accuracy (better than random forest)
    base_scores2 = cross_val_score(model2, X_train, y_train, cv=4, scoring="accuracy")
    print("Cross validation score:", base_scores2)
    print(f"Mean validation score: {np.mean(base_scores2):.4f}")
    
    
    # Use GridSearchCV to find best hyperparameter combinations for chosen model (Logistic Regressin)
    grid = GridSearchCV(model2, params2, cv=4, scoring="accuracy", n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)
    print(f"Tuned mean accuracy: {grid.best_score_:.3f}")
    # Best parameters: {'C': 1.0, 'max_iter': 100, 'solver': 'lbfgs'}
    # The tuned mean accuracy: ~86%
    
    
  
if __name__ == "__main__":
    main()
    