from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
from load_disease_data import load_dataset
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RandomForest

@st.cache_resource(show_spinner=False)
def load_model():
    # script_dir = Path(__file__).parent
    # model_path = script_dir / "models" / "best_random_forest.joblib"

    # Check if model is in path
    # if not model_path.exists():
    #     st.error(
    #         f"Model file not found at {model_path}"
    #     )
    #     st.stop()

    # Load our model in with joblib
    # model = joblib.load(model_path)
    # Load in our full dataset
    X, y = load_dataset(use_full_data=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForest(
        random_state = 42,
        criterion = 'gini',
        max_depth = None,
        max_features = "log2",
        min_samples_leaf = 2,
        n_estimators = 100
    )
    model.fit(X_train, y_train)

    return model, X_train

# Process user's input to match the format that the model expects


def process_user_input(user_data: list, column_names: list) -> pd.DataFrame:
    user_df = pd.DataFrame([user_data], columns=column_names)
    return user_df


def main() -> None:
    with open("data/raw/description.json", "r") as file:
        data_dict = json.load(file)

    st.set_page_config(
        page_title="Medical Symptom Analyzer",
        page_icon="🩺",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    st.title("🩺 Medical Symptom Analyzer")
    st.caption("AI powered preliminary illness assessment based on reported symptoms")
    st.info(
    "⚕️ **Medical Disclaimer:** This tool is for informational purposes only and does not replace professional medical advice. "
    "Please consult a healthcare professional for an accurate diagnosis and treatment."
    )
    with st.expander("How to Use This Tool", expanded=False):
        st.markdown(
            """
            1. **Select Symptoms**: Use the search box below to find and select your current symptoms
            2. **Review Selection**: Verify you've selected all your relevant symptoms
            3. **Get Results**: Click the analyze button to receive predictions
            4. **Interpret Results**: Review the top 3 most likely conditions
            """
        )
    
    model, X_train = load_model()
    
    # Adjust symptom names for searchability
    all_symptoms = sorted([col.replace('_', " ").title() for col in X_train.columns])

    # Multi select for all symptoms
    selected_symptoms = st.multiselect(
        "Select all symptoms you are experiencing:",
        options=all_symptoms,
        placeholder="Type to search symptoms (e.g., 'fever', 'cough', 'headache')",
        help="You can select multiple symptoms, the more accurate your selection, the more accurate the prediction."
    )

    # Show selected symptoms
    if selected_symptoms:
        st.success(f"✅ {len(selected_symptoms)} symptom(s) selected")
        with st.expander("📋 View Selected Symptoms"):
            for symptom in selected_symptoms:
                st.write(f"• {symptom}")
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🔍 Analyze Symptoms",
            use_container_width=True,
            type="primary",
            disabled=(len(selected_symptoms) == 0)
        )

    if analyze_button:
        if len(selected_symptoms) == 0:
            st.error("Select at least one symptom")
        else:
            # Set all symptoms to 0 (False)
            symptom_list = [0] * len(X_train.columns)
            # Iterate through and change symptom to 1 (True) if user has it
            for symptom in selected_symptoms:
                name = symptom.lower().replace(' ', '_')  # Convert back    
                if name in X_train.columns:
                    idx = X_train.columns.get_loc(name)
                    symptom_list[idx] = 1

        user_data = process_user_input(symptom_list, X_train.columns)
        prediction = model.predict_proba(user_data)[0]
        top_3_symp_idx = prediction.argsort()[-3:][::-1]   
        
        st.divider()
        st.subheader("Analysis Results")
        st.subheader("Top 3 potential conditions")
        for i, idx in enumerate(top_3_symp_idx, 1):
            disease = model.classes_[idx]
            confidence = prediction[idx] * 100
            # Convert to format for JSON
            disease_key = disease.split('(')[0].strip()
            disease_key = disease_key.lower().replace(' ', '_')
            
            # Access info through key
            disease_info = data_dict[disease_key]
            # Display top 3 diseases to user
            st.markdown(f"### {disease_info['name']}")
            st.progress(confidence / 100)
            st.caption(f"Confidence: {confidence:.1f}%")
            st.markdown("**Description:**")
            st.write(disease_info['description'])
            st.markdown("**Severity Level:**")
            severity = disease_info['severity']
            # Display severity and guide user
            if severity.lower() in ['high', 'severe', 'critical']:
                st.error(f"🔴 {severity}")
                st.caption("Seek immediate medical attention")
            elif severity.lower() in ['medium', 'moderate']:
                st.warning(f"🟡 {severity}")
                st.caption("Consult a doctor soon")
            else:
                st.info(f"🟢 {severity}")
                st.caption("Monitor symptoms")

if __name__ == "__main__":
    main()
