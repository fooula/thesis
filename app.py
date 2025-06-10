import streamlit as st
import pandas as pd
import joblib

# Φόρτωση μοντέλου και χαρακτηριστικών
model = joblib.load("xgboost_model.pkl")
model_features = joblib.load("xgboost_model.pkl")

# Mapping dictionaries
sex_map = {"Άνδρας": 0, "Γυναίκα": 1}
treatment_map = {"Συντηρητική": 0, "Χειρουργική": 1}
physio_map = {"Όχι": 0, "Ναι": 1}
fracture_map = {"Απλό": 0, "Σύνθετο": 1}

def prepare_input_data(age, sex, treatment_type, early_physiotherapy, osteoporosis,
                       diabetes, fracture_type, physio_sessions,
                       grip_strength_improvement, dash_score_6months=20.0,
                       rom_extension_3m=45.0, rom_flexion_3m=50.0,
                       rom_supination_3m=60.0, rom_pronation_3m=55.0):
    input_data = pd.DataFrame([[ 
        age,
        sex_map[sex],
        treatment_map[treatment_type],
        physio_map[early_physiotherapy],
        physio_map[osteoporosis],
        physio_map[diabetes],
        fracture_map[fracture_type],
        physio_sessions,
        grip_strength_improvement,
        dash_score_6months,
        rom_extension_3m,
        rom_flexion_3m,
        rom_supination_3m,
        rom_pronation_3m
    ]], columns=[
        'age', 'sex', 'treatment_type', 'early_physiotherapy',
        'osteoporosis', 'diabetes', 'fracture_type',
        'physio_sessions', 'grip_strength_improvement',
        'dash_score_6months', 'rom_extension_3m', 'rom_flexion_3m',
        'rom_supination_3m', 'rom_pronation_3m'
    ])

    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[model_features]
    return input_data

# Streamlit UI
st.title("🦴 Εκτίμηση Χρόνου Αποκατάστασης μετά από Κάταγμα Κερκίδας")

st.subheader("📋 Εισαγωγή Στοιχείων Ασθενούς")
age = st.slider("Ηλικία", 18, 100, 40)
sex = st.selectbox("Φύλο", list(sex_map.keys()))
treatment_type = st.selectbox("Τύπος Αντιμετώπισης", list(treatment_map.keys()))
early_physiotherapy = st.selectbox("Έναρξη Φυσικοθεραπείας <2 Εβδομάδες", list(physio_map.keys()))
osteoporosis = st.selectbox("Οστεοπόρωση", list(physio_map.keys()))
diabetes = st.selectbox("Διαβήτης", list(physio_map.keys()))
fracture_type = st.selectbox("Τύπος Κατάγματος", list(fracture_map.keys()))
physio_sessions = st.slider("Συνεδρίες Φυσικοθεραπείας", 0, 50, 10)
grip_strength_improvement = st.slider("Βελτίωση Λαβής (%)", 0, 100, 30)

if st.button("📈 Υπολογισμός Χρόνου Αποκατάστασης"):
    input_df = prepare_input_data(
        age, sex, treatment_type, early_physiotherapy,
        osteoporosis, diabetes, fracture_type,
        physio_sessions, grip_strength_improvement
    )
    prediction = model.predict(input_df)[0]
    st.success(f"📅 Εκτιμώμενος Χρόνος Αποκατάστασης: **{prediction:.2f} εβδομάδες**")

st.markdown("---")
st.subheader("🧪 What-if Ανάλυση")
if st.button("🔍 Τρέξε What-if Πρόβλεψη"):
    input_df = prepare_input_data(
        age, sex, treatment_type, early_physiotherapy,
        osteoporosis, diabetes, fracture_type,
        physio_sessions, grip_strength_improvement
    )
    prediction = model.predict(input_df)[0]
    st.success(f"📅 What-if Εκτίμηση: **{prediction:.2f} εβδομάδες**")
