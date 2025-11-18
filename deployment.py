from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from exploratory_analysis import load_dataset
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_model():
    script_dir = Path(__file__).parent
    model_path = script_dir / "models" / "best_logistic_regression.joblib"
    
    # Check if model is in path
    if not model_path.exists():
        st.error(
            f"Model file not found at {model_path}"
        )
        st.stop()
    
    # Load our model in with joblib
    model = joblib.load(model_path)
    # Load in our full dataset
    X, y = load_dataset(use_full_data=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return model, X_train
    