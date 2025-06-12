import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Ρύθμιση σελίδας
st.set_page_config(page_title="Εκτίμηση Χρόνου Αποκατάστασης", layout="centered")

# Φόρτωση μοντέλου και δεδομένων
model = joblib.load("xgboost_model.pkl")
df = pd.read_csv("distal_radius_recovery.csv")

# Προειδοποίηση για συνθετικά δεδομένα
st.warning(" Τα δεδομένα είναι συνθετικά και χρησιμοποιούνται μόνο για επίδειξη.", icon="⚠️")

# Τίτλος
st.title("🦴 Πρόβλεψη Χρόνου Αποκατάστασης από Κάταγμα Κερκίδας")

# Sidebar για είσοδο χρήστη
st.sidebar.header("🔍 Εισαγωγή χαρακτηριστικών ασθενή")

def user_input_features():
    age = st.sidebar.slider("Ηλικία", 18, 90, 45)
    sex = st.sidebar.selectbox("Φύλο", ["Άνδρας", "Γυναίκα"])
    treatment_type = st.sidebar.selectbox("Μέθοδος Θεραπείας", ["Συντηρητική", "Χειρουργική"])
    early_physio = st.sidebar.selectbox("Έναρξη φυσικοθεραπείας <14 ημέρες", ["Όχι", "Ναι"])
    osteoporosis = st.sidebar.selectbox("Οστεοπόρωση", ["Όχι", "Ναι"])
    diabetes = st.sidebar.selectbox("Διαβήτης", ["Όχι", "Ναι"])
    fracture_type = st.sidebar.selectbox("Τύπος κατάγματος", ["Τύπος A", "Τύπος B", "Τύπος C"])
    physio_sessions = st.sidebar.slider("Συνεδρίες φυσικοθεραπείας", 0, 30, 10)
    grip = st.sidebar.slider("Βελτίωση δύναμης λαβής (%)", 0, 100, 50)
    dash = st.sidebar.slider("DASH score (6 μήνες)", 0.0, 100.0, 50.0)
    rom_ext = st.sidebar.slider("Έκταση ROM (3 μήνες)", 0.0, 100.0, 50.0)
    rom_flex = st.sidebar.slider("Κάμψη ROM (3 μήνες)", 0.0, 100.0, 50.0)
    rom_sup = st.sidebar.slider("Υπτιασμός ROM (3 μήνες)", 0.0, 100.0, 50.0)
    rom_pro = st.sidebar.slider("Πρηνισμός ROM (3 μήνες)", 0.0, 100.0, 50.0)
    age_group = st.sidebar.selectbox("Ηλικιακή Ομάδα", ["<65", "65-75", ">75"])
    risk_triad = st.sidebar.selectbox("Κατηγορία Κινδύνου", ["Χαμηλός", "Μέτριος", "Υψηλός"])
    cci = st.sidebar.slider("Δείκτης Charlson", 0, 10, 1)
    edmonton = st.sidebar.slider("Edmonton Frail Scale", 0, 17, 5)
    pase = st.sidebar.slider("PASE Score", 0, 400, 150)
    displacement = st.sidebar.selectbox("Μετατόπιση κατάγματος", ["Όχι", "Ναι"])
    stability = st.sidebar.selectbox("Σταθερότητα κατάγματος", ["Σταθερό", "Ασταθές"])

    data = {
        "age": age,
        "sex": 1 if sex == "Άνδρας" else 0,
        "treatment_type": 1 if treatment_type == "Χειρουργική" else 0,
        "early_physiotherapy": 1 if early_physio == "Ναι" else 0,
        "osteoporosis": 1 if osteoporosis == "Ναι" else 0,
        "diabetes": 1 if diabetes == "Ναι" else 0,
        "fracture_type": {"Τύπος A": 0, "Τύπος B": 1, "Τύπος C": 2}[fracture_type],
        "physio_sessions": physio_sessions,
        "grip_strength_improvement": grip,
        "dash_score_6months": dash,
        "rom_extension_3m": rom_ext,
        "rom_flexion_3m": rom_flex,
        "rom_supination_3m": rom_sup,
        "rom_pronation_3m": rom_pro,
        "age_group": {"<65": 0, "65-75": 1, ">75": 2}[age_group],
        "risk_triad": {"Χαμηλός": 0, "Μέτριος": 1, "Υψηλός": 2}[risk_triad],
        "charlson_index": cci,
        "edmonton_frail_scale": edmonton,
        "pase_score": pase,
        "displacement": 1 if displacement == "Ναι" else 0,
        "fracture_stability": 1 if stability == "Ασταθές" else 0,
    }

    return pd.DataFrame([data])

input_df = user_input_features()

# Κουμπί πρόβλεψης
if st.button("🔮 Υπολογισμός Χρόνου Αποκατάστασης"):
    prediction = model.predict(input_df)[0]
    st.subheader("🧾 Εκτίμηση:")
    st.success(f"Εκτιμώμενος χρόνος αποκατάστασης: **{int(round(prediction))} εβδομαδες**")

    # ➕ Σύγκριση με παρόμοιες περιπτώσεις
    similar = df[
        (df["age_group"] == input_df["age_group"].values[0]) &
        (df["fracture_type"] == input_df["fracture_type"].values[0]) &
        (df["treatment_type"] == input_df["treatment_type"].values[0])
    ]

    if not similar.empty:
        avg_days = similar["recovery_days"].mean()
        count = similar.shape[0]
        st.info(f"🔍 Βρέθηκαν **{count}** παρόμοιες περιπτώσεις. "
                f"Μέσος χρόνος αποκατάστασης: **{int(round(avg_days))} εβδομαδες**")

        # ➕ Γράφημα
        fig, ax = plt.subplots()
        ax.hist(similar["recovery_days"], bins=15, color="skyblue", edgecolor="black")
        ax.axvline(prediction, color="red", linestyle="--", label="Η πρόβλεψή σου")
        ax.set_title("Κατανομή Χρόνου Αποκατάστασης σε Παρόμοιες Περιπτώσεις")
        ax.set_xlabel("Εβδομαδες")
        ax.set_ylabel("Αριθμός Ασθενών")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("❌ Δεν βρέθηκαν αρκετά παρόμοια περιστατικά για σύγκριση.")

# Προβολή δείγματος δεδομένων
if st.checkbox("📁 Δείγμα συνθετικών δεδομένων"):
    st.dataframe(df.sample(5))
