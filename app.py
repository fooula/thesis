import streamlit as st
import pandas as pd
import joblib

# Ρύθμιση σελίδας
st.set_page_config(page_title="Εκτίμηση Χρόνου Αποκατάστασης", layout="centered")

# Φόρτωση μοντέλου (που έχει εκπαιδευτεί σε ακατέργαστα δεδομένα)
model = joblib.load("xgboost_model.pkl")

# Φόρτωση dataset (προαιρετικά για προβολή)
df = pd.read_csv("distal_radius_recovery.csv")
st.warning(" Τα δεδομένα είναι συνθετικά και χρησιμοποιούνται μόνο για επίδειξη.", icon="⚠️")

# Τίτλος
st.title("🦴 Πρόβλεψη Χρόνου Αποκατάστασης από Κάταγμα Κερκίδας")

# Περιγραφή
st.markdown("""
Εισάγετε τα χαρακτηριστικά του ασθενούς και της κατάστασης για να προβλεφθεί ο εκτιμώμενος χρόνος αποκατάστασης (σε εβδομαδες).
""")

# Sidebar για είσοδο χαρακτηριστικών
def user_input_features():
    age = st.sidebar.slider("Ηλικία", 18, 90, 45)
    sex = st.sidebar.selectbox("Φύλο", ["Άνδρας", "Γυναίκα"])
    treatment_type = st.sidebar.selectbox("Τύπος Θεραπείας", ["Συντηρητική", "Χειρουργική"])
    early_physiotherapy = st.sidebar.selectbox("Έναρξη Φυσικοθεραπείας νωρίς;", ["Ναι", "Όχι"])
    osteoporosis = st.sidebar.selectbox("Οστεοπόρωση;", ["Ναι", "Όχι"])
    diabetes = st.sidebar.selectbox("Διαβήτης;", ["Ναι", "Όχι"])
    fracture_type = st.sidebar.selectbox("Τύπος Κατάγματος", ["A", "B", "C"])
    physio_sessions = st.sidebar.slider("Συνεδρίες Φυσικοθεραπείας", 0, 30, 10)
    grip_strength_improvement = st.sidebar.slider("Βελτίωση Δύναμης Χειρός (%)", 0, 100, 50)
    dash_score_6months = st.sidebar.slider("DASH Score (6 μήνες)", 0, 100, 20)
    rom_extension_3m = st.sidebar.slider("ROM Έκταση (3 μήνες)", 0, 150, 120)
    rom_flexion_3m = st.sidebar.slider("ROM Κάμψη (3 μήνες)", 0, 150, 130)
    rom_supination_3m = st.sidebar.slider("ROM Υπτιασμός (3 μήνες)", 0, 150, 90)
    rom_pronation_3m = st.sidebar.slider("ROM Προσποίηση (3 μήνες)", 0, 150, 90)
    age_group = st.sidebar.selectbox("Ομάδα Ηλικίας", ["Νεαρός", "Μέσης Ηλικίας", "Ηλικιωμένος"])
    risk_triad = st.sidebar.selectbox("Risk Triad", ["Χαμηλός", "Μέτριος", "Υψηλός"])
    charlson_index = st.sidebar.slider("Charlson Index", 0, 10, 2)
    edmonton_frail_scale = st.sidebar.slider("Edmonton Frail Scale", 0, 10, 3)
    pase_score = st.sidebar.slider("PASE Score", 0, 400, 150)
    displacement = st.sidebar.selectbox("Μετατόπιση Κατάγματος", ["Ναι", "Όχι"])
    fracture_stability = st.sidebar.selectbox("Σταθερότητα Κατάγματος", ["Ναι", "Όχι"])

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
        "age_group": {"Νεαρός": 0, "Μέσης Ηλικίας": 1, "Ηλικιωμένος": 2}[age_group],
        "risk_triad": {"Χαμηλός": 0, "Μέτριος": 1, "Υψηλός": 2}[risk_triad],
        "charlson_index": charlson_index,
        "edmonton_frail_scale": edmonton_frail_scale,
        "pase_score": pase_score,
        "displacement": 1 if displacement == "Ναι" else 0,
        "fracture_stability": 1 if fracture_stability == "Ναι" else 0
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Καταχώρηση των στηλών που περιμένει το μοντέλο με ακριβή σειρά
model_features = [
    "age", "sex", "treatment_type", "early_physiotherapy", "osteoporosis", "diabetes",
    "fracture_type", "physio_sessions", "grip_strength_improvement", "dash_score_6months",
    "rom_extension_3m", "rom_flexion_3m", "rom_supination_3m", "rom_pronation_3m",
    "age_group", "risk_triad", "charlson_index", "edmonton_frail_scale", "pase_score",
    "displacement", "fracture_stability"
]

input_df = input_df[model_features]  # Βεβαιώσου ότι η σειρά στηλών ταιριάζει

# Πρόβλεψη (χωρίς scaler)
if st.button("🔮 Υπολογισμός Χρόνου Αποκατάστασης"):
    prediction = model.predict(input_df)[0]
    st.subheader("🧾 Εκτίμηση:")
    st.success(f"Εκτιμώμενος χρόνος αποκατάστασης: **{int(round(prediction))} εβδομαδες**")

# Προβολή δείγματος συνθετικών δεδομένων
if st.checkbox("📁 Δείγμα συνθετικών δεδομένων"):
    st.dataframe(df.sample(5))

