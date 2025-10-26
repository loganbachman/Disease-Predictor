import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from kaggle.api.kaggle_api_extended import KaggleApi #type: ignore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns #type: ignore
from sklearn.ensemble import RandomForestClassifier

api = KaggleApi()
api.authenticate()

def main():
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
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=25000, stratify=y, random_state=42)
    
    # filter out diseases that dont appear at least 5 times
    disease_counts = y_sample.value_counts()
    valid_diseases = disease_counts[disease_counts >= 10].index
    valid = y_sample.isin(valid_diseases)
    
    # assign X and y to filtered diseases
    X_sample = X_sample[valid]
    y_sample = y_sample[valid]
    
    # Split sample into training and testing set (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, random_state=42, test_size=0.20)
    
    # Hyperparameter Tuning 
    # def tune_model (X_train, y_train)
     #   param_grid 




    # Assign model to random forest with 100 trees
    model = RandomForestClassifier(n_estimators=100)
    
    # Cross-fold validation and score
    kf = KFold(n_splits=4)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="accuracy")
    print("Cross validation score:", cv_scores)
    print(f"Mean validation score: {np.mean(cv_scores):.4f}")
    
    
  
if __name__ == "__main__":
    main()
    