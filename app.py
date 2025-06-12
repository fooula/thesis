import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb

# Τίτλος
st.title("Εκτίμηση Χρόνου Αποκατάστασης Κατάγματος Κερκίδας")

# ----------------------------------------
# Φόρτωση εκπαιδευμένου μοντέλου
# ----------------------------------------
with open("xgboost_model.pkl", "rb") as file:
    model = pickle.load(file)

# ----------------------------------------
# Συνάρτηση για είσοδο δεδομένων από χρήστη
# ----------------------------------------
def user_input_features():
    age = st.sidebar.slider("Ηλικία", 18, 90, 50)
    sex = st.sidebar.selectbox("Φύλο", ["Άνδρας", "Γυναίκα"])
    treatment_type = st.sidebar.selectbox("Τύπος Θεραπείας", ["Συντηρητική", "Χειρουργική"])
    early_physiotherapy = st.sidebar.selectbox("Έγκαιρη φυσικοθεραπεία;", ["Ναι", "Όχι"])
    osteoporosis = st.sidebar.selectbox("Οστεοπόρωση;", ["Ναι", "Όχι"])
    diabetes = st.sidebar.selectbox("Διαβήτης;", ["Ναι", "Όχι"])
    fracture_type = st.sidebar.selectbox("Τύπος Κατάγματος", ["A", "B", "C"])
    physio_sessions = st.sidebar.slider("Συνεδρίες Φυσικοθεραπείας", 0, 30, 10)
    grip_strength_improvement = st.sidebar.slider("Βελτίωση δύναμης λαβής (%)", 0, 100, 50)
    dash_score_6months = st.sidebar.slider("DASH Score (6 μήνες)", 0, 100, 30)
    rom_extension_3m = st.sidebar.slider("ROM - Έκταση (3 μήνες)", 0, 100, 70)
    rom_flexion_3m = st.sidebar.slider("ROM - Κάμψη (3 μήνες)", 0, 100, 75)
    rom_supination_3m = st.sidebar.slider("ROM - Υπτιασμός (3 μήνες)", 0, 100, 80)
    rom_pronation_3m = st.sidebar.slider("ROM - Πρηνισμός (3 μήνες)", 0, 100, 80)
    age_group = st.sidebar.selectbox("Ηλικιακή Ομάδα", ["<40", "40-65", ">65"])
    risk_triad = st.sidebar.selectbox("Κίνδυνος (Triad)", ["Χαμηλός", "Μέτριος", "Υψηλός"])
    charlson_index = st.sidebar.slider("Charlson Comorbidity Index", 0, 10, 2)
    edmonton_frail_scale = st.sidebar.slider("Edmonton Frail Scale", 0, 17, 5)
    pase_score = st.sidebar.slider("PASE Score", 0, 400, 100)
    displacement = st.sidebar.selectbox("Μετατόπιση κατάγματος;", ["Όχι", "Ναι"])
    fracture_stability = st.sidebar.selectbox("Σταθερότητα Κατάγματος", ["Σταθερό", "Ασταθές"])

    data = {
        "age": age,
        "sex": 1 if sex == "Άνδρας" else 0,
        "treatment_type": 1 if treatment_type == "Χειρουργική" else 0,
        "early_physiotherapy": 1 if early_physiotherapy == "Ναι" else 0,
        "osteoporosis": 1 if osteoporosis == "Ναι" else 0,
        "diabetes": 1 if diabetes == "Ναι" else 0,
        "fracture_type": {"A": 0, "B": 1, "C": 2}[fracture_type],
        "physio_sessions": physio_sessions,
        "grip_strength_improvement": grip_strength_improvement,
        "dash_score_6months": dash_score_6months,
        "rom_extension_3m": rom_extension_3m,
        "rom_flexion_3m": rom_flexion_3m,
        "rom_supination_3m": rom_supination_3m,
        "rom_pronation_3m": rom_pronation_3m,
        "age_group": {"<40": 0, "40-65": 1, ">65": 2}[age_group],
        "risk_triad": {"Χαμηλός": 0, "Μέτριος": 1, "Υψηλός": 2}[risk_triad],
        "charlson_index": charlson_index,
        "edmonton_frail_scale": edmonton_frail_scale,
        "pase_score": pase_score,
        "displacement": 1 if displacement == "Ναι" else 0,
        "fracture_stability": 1 if fracture_stability == "Ασταθές" else 0
    }

    return pd.DataFrame([data])

# ----------------------------------------
# Δημιουργία input dataframe από χρήστη
# ----------------------------------------


input_df = user_input_features()
st.write("Model expects features:")
st.write(model.get_booster().feature_names)
st.write("Input DataFrame columns:")
st.write(input_df.columns.tolist())
# Διάταξη input_df με τα ίδια features όπως το μοντέλο
model_features = model.get_booster().feature_names
input_df = input_df[model_features]

# ----------------------------------------
# Πρόβλεψη
# ----------------------------------------
prediction = model.predict(input_df)[0]

# ----------------------------------------
# Εμφάνιση αποτελέσματος
# ----------------------------------------
st.subheader("Εκτιμώμενος Χρόνος Αποκατάστασης (σε εβδομάδες):")
st.success(f"{prediction:.1f} εβδομάδες")

st.subheader("Είσοδοι Χρήστη")
st.write(input_df)
