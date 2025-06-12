import streamlit as st
import pandas as pd
import pickle

# Φόρτωση εκπαιδευμένου μοντέλου
with open("xgboost_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Εκτίμηση Χρόνου Αποκατάστασης Κατάγματος Κερκίδας")

# ==== Φόρμα εισόδου χρήστη ====
age = st.number_input("Ηλικία", min_value=0, max_value=120, value=65)
sex = st.selectbox("Φύλο", ["Άνδρας", "Γυναίκα"])
treatment_type = st.selectbox("Τύπος θεραπείας", ["Χειρουργική", "Συντηρητική"])
early_physiotherapy = st.selectbox("Έναρξη φυσικοθεραπείας <2 εβδομάδες;", ["Ναι", "Όχι"])
osteoporosis = st.selectbox("Οστεοπόρωση", ["Ναι", "Όχι"])
diabetes = st.selectbox("Σακχαρώδης Διαβήτης", ["Ναι", "Όχι"])
fracture_type = st.selectbox("Τύπος κατάγματος", ["Έγκλειστο", "Ανοικτό"])
physio_sessions = st.number_input("Αριθμός φυσικοθεραπειών", min_value=0, value=10)
grip_strength_improvement = st.number_input("Βελτίωση δύναμης λαβής (%)", min_value=0.0, max_value=100.0, value=20.0)
dash_score_6months = st.number_input("DASH score στους 6 μήνες", min_value=0.0, max_value=100.0, value=30.0)
rom_extension_3m = st.number_input("ROM έκτασης 3 μηνών (°)", value=50.0)
rom_flexion_3m = st.number_input("ROM κάμψης 3 μηνών (°)", value=60.0)
rom_supination_3m = st.number_input("ROM υπτιασμού 3 μηνών (°)", value=45.0)
rom_pronation_3m = st.number_input("ROM πρηνισμού 3 μηνών (°)", value=45.0)
age_group = st.selectbox("Ηλικιακή ομάδα", ["<65", "65-80", ">80"])
risk_triad = st.selectbox("Κίνδυνος (Risk Triad)", ["Χαμηλός", "Μέτριος", "Υψηλός"])
charlson_index = st.number_input("Δείκτης Charlson", min_value=0, max_value=20, value=2)
edmonton_frail_scale = st.number_input("Edmonton Frail Scale", min_value=0, max_value=17, value=5)
pase_score = st.number_input("PASE score", min_value=0, value=120)
displacement = st.selectbox("Μετατόπιση κατάγματος;", ["Ναι", "Όχι"])
fracture_stability = st.selectbox("Σταθερότητα κατάγματος;", ["Σταθερό", "Ασταθές"])

# ==== Κωδικοποίηση τιμών ====
input_data = {
    "age": age,
    "sex": 0 if sex == "Άνδρας" else 1,
    "treatment_type": 0 if treatment_type == "Χειρουργική" else 1,
    "early_physiotherapy": 1 if early_physiotherapy == "Ναι" else 0,
    "osteoporosis": 1 if osteoporosis == "Ναι" else 0,
    "diabetes": 1 if diabetes == "Ναι" else 0,
    "fracture_type": 0 if fracture_type == "Έγκλειστο" else 1,
    "physio_sessions": physio_sessions,
    "grip_strength_improvement": grip_strength_improvement,
    "dash_score_6months": dash_score_6months,
    "rom_extension_3m": rom_extension_3m,
    "rom_flexion_3m": rom_flexion_3m,
    "rom_supination_3m": rom_supination_3m,
    "rom_pronation_3m": rom_pronation_3m,
    "age_group": {"<65": 0, "65-80": 1, ">80": 2}[age_group],
    "risk_triad": {"Χαμηλός": 0, "Μέτριος": 1, "Υψηλός": 2}[risk_triad],
    "charlson_index": charlson_index,
    "edmonton_frail_scale": edmonton_frail_scale,
    "pase_score": pase_score,
    "displacement": 1 if displacement == "Ναι" else 0,
    "fracture_stability": 0 if fracture_stability == "Σταθερό" else 1,
}

# ==== Δημιουργία DataFrame ====
input_df = pd.DataFrame([input_data])

# ==== Πρόβλεψη ====
if st.button("Πρόβλεψη Χρόνου Αποκατάστασης"):
    prediction = model.predict(input_df)[0]
    st.success(f"Εκτιμώμενος χρόνος αποκατάστασης: {prediction:.1f} ημέρες")
