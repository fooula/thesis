import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt  # Αν δεν το χρησιμοποιείς αλλού, μπορείς να το αφαιρέσεις και αυτό

# Σελίδα Streamlit
st.set_page_config(page_title="Εκτίμηση Χρόνου Αποκατάστασης", layout="centered")

# Εισαγωγή μοντέλου
model = joblib.load("xgboost_model.pkl")

# Εισαγωγή δεδομένων
df = pd.read_csv("distal_radius_recovery.csv")

# Ενημέρωση για συνθετικά δεδομένα
st.warning("⚠️ Τα δεδομένα είναι συνθετικά και χρησιμοποιούνται μόνο για επίδειξη.", icon="⚠️")

# Τίτλος
st.title("🦴 Πρόβλεψη Χρόνου Αποκατάστασης από Κάταγμα Κερκίδας")

# Περιγραφή
st.markdown("""
Η εφαρμογή αυτή προβλέπει τον εκτιμώμενο χρόνο αποκατάστασης (σε ημέρες) από κάταγμα κερκίδας, 
βασισμένη σε χαρακτηριστικά του ασθενή και του τραυματισμού.
""")

# Sidebar για είσοδο δεδομένων χρήστη
st.sidebar.header("🔍 Εισαγωγή χαρακτηριστικών ασθενή")

def user_input_features():
    age = st.sidebar.slider("Ηλικία", 18, 90, 45)
    gender = st.sidebar.selectbox("Φύλο", ["Άνδρας", "Γυναίκα"])
    weight = st.sidebar.slider("Βάρος (kg)", 40, 120, 70)
    fracture_type = st.sidebar.selectbox("Τύπος Κατάγματος", ["Τύπος A", "Τύπος B", "Τύπος C"])
    treatment = st.sidebar.selectbox("Μέθοδος Θεραπείας", ["Συντηρητική", "Χειρουργική"])
    physiotherapy_sessions = st.sidebar.slider("Αριθμός Συνεδριών Φυσικοθεραπείας", 0, 30, 10)
    smoking = st.sidebar.selectbox("Καπνίζει;", ["Όχι", "Ναι"])
    diabetes = st.sidebar.selectbox("Διαβήτης;", ["Όχι", "Ναι"])

    data = {
        "Age": age,
        "Gender": 1 if gender == "Άνδρας" else 0,
        "Weight": weight,
        "Fracture_Type": {"Τύπος A": 0, "Τύπος B": 1, "Τύπος C": 2}[fracture_type],
        "Treatment_Type": 1 if treatment == "Χειρουργική" else 0,
        "Physiotherapy_Sessions": physiotherapy_sessions,
        "Smoking": 1 if smoking == "Ναι" else 0,
        "Diabetes": 1 if diabetes == "Ναι" else 0
    }

    return pd.DataFrame([data])

input_df = user_input_features()

# Πρόβλεψη
if st.button("🔮 Υπολογισμός Χρόνου Αποκατάστασης"):
    prediction = model.predict(input_df)[0]
    st.subheader("🧾 Εκτίμηση:")
    st.success(f"Εκτιμώμενος χρόνος αποκατάστασης: **{int(round(prediction))} ημέρες**")

# Προβολή δείγματος δεδομένων
if st.checkbox("📁 Δείγμα συνθετικών δεδομένων"):
    st.dataframe(df.sample(5))
