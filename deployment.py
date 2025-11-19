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

# Process user's input to match the format that the model expects
def process_user_input(user_data: dict) -> NDArray:
    user_df = pd.DataFrame([user_data])
    
    return user_df

def main() -> None:
    st.set_page_config(
        page_title="Illness Prediction",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Patient Illness Predictor")
    st.markdown(
        """
        This application will predict someone's likely illnesses based on their
        reported symptoms. The model was trained on reported patient symptoms and
        can predict what illness the patient is experiencing
        
        Fill out the symptoms below and click predict to get a diagnosis!
        """
    )
    
    model, X_train = load_model()
    # Adjust symptom names for searchability
    all_symptoms = sorted([col.replace('_', " ").title() for col in X_train.columns])
    
    # Form to collect user input
    with st.form("symptom_form"):
        st.header("Patient Symptoms")
        
        # Multi select for all symptoms
        selected_symptoms = st.multiselect(
            "Symptoms",
            options=all_symptoms,
            placeholder="Type to search symptoms (e.g., 'fever', 'cough', 'headache')",
            help="Select all symptoms that you're experiencing"
        )
        
        # Submit button
        submitted = st.form_submit_button(
            "Predict your potential illnesses", use_container_width=True
        )
        
        if selected_symptoms:
            st.info(f"✓ {len(selected_symptoms)} symptoms selected")
        
    if submitted:
        if len(selected_symptoms) == 0:
            st.error("Select at least one symptom")
        else:
            # Set all symptoms to 0 (False)
            symptom_list = [0] * len(X_train.columns)
            # Iterate through and change symptom to 1 (True) if user has it
            for symptom in selected_symptoms:
                name = symptom.lower().replace(' ', '_') # Convert back to original name
                
                if name in X_train.columns:
                    idx = X_train.columns.get_loc(name)
                    symptom_list[idx] = 1
                
        
    