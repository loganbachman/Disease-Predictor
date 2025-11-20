from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
from load_disease_data import load_dataset
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
def process_user_input(user_data: list, column_names: list) -> pd.DataFrame:
    user_df = pd.DataFrame([user_data], columns=column_names)
    return user_df

def main() -> None:
    with open("data/raw/description.json", "r") as file:
        data_dict = json.load(file)
    
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
        
        user_data = process_user_input(symptom_list, X_train.columns)
        prediction = model.predict_proba(user_data)[0]
        top_3_symp_idx = prediction.argsort()[-3:][::-1]
        
        st.subheader("Top 3 Predicted Illnesses")
        for i, idx in enumerate(top_3_symp_idx, 1):
            disease = model.classes_[idx]
            confidence = prediction[idx] * 100
            # Convert to format for JSON
            disease_key = disease.split('(')[0].strip()
            disease_key = disease_key.lower().replace(' ', '_')
            
            # Access info through key
            disease_info = data_dict[disease_key]
            # Display top 3 diseases to user
            st.write(
                f"""
                {i}. **{disease_info['name']}** - {confidence:.1f}% Confidence\n
                **Info** - {disease_info['description']}\n
                **Severity** - {disease_info['severity']}
                """
            )






if __name__ == "__main__":
    main()
                
        
    