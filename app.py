import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Φόρτωση μοντέλου και δεδομένων
model = joblib.load("xgboost_model.pkl")
df = pd.read_csv("distal_radius_recovery.csv")

model_features = [
    "age", "sex", "treatment_type", "early_physiotherapy", "osteoporosis",
    "diabetes", "fracture_type", "physio_sessions", "grip_strength_improvement",
    "dash_score_6months", "rom_extension_3m", "rom_flexion_3m",
    "rom_supination_3m", "rom_pronation_3m", "age_group", "risk_triad",
    "charlson_index", "edmonton_frail_scale", "pase_score",
    "displacement", "fracture_stability"
]

st.title("🦴 Πρόβλεψη Χρόνου Αποκατάστασης Κατάγματος Κερκίδας")

# Επιλογή τύπου χρήστη
user_type = st.sidebar.selectbox("Επιλέξτε Χρήστη", ["Ασθενής", "Γιατρός / Ειδικός"])

st.sidebar.header("🔢 Εισαγωγή στοιχείων ασθενούς")

# Πεδία που θα εμφανίζονται σε όλους
age = st.sidebar.number_input("Ηλικία", min_value=18, max_value=100, value=50)
sex = st.sidebar.selectbox("Φύλο", ["Άνδρας", "Γυναίκα"])
treatment_type = st.sidebar.selectbox("Τύπος θεραπείας", ["Συντηρητική", "Χειρουργική"])
early_physiotherapy = st.sidebar.selectbox("Έναρξη φυσιοθεραπείας εντός 2 εβδομάδων;", ["Όχι", "Ναι"])

# Περισσότερα πεδία μόνο για Γιατρό/Ειδικό
if user_type == "Γιατρός / Ειδικός":
    osteoporosis = st.sidebar.selectbox("Οστεοπόρωση", ["Όχι", "Ναι"])
    diabetes = st.sidebar.selectbox("Σακχαρώδης Διαβήτης", ["Όχι", "Ναι"])
    fracture_type = st.sidebar.selectbox("Τύπος κατάγματος", ["Απλό", "Σύνθετο"])
    physio_sessions = st.sidebar.number_input("Αριθμός συνεδριών φυσιοθεραπείας", min_value=0, max_value=100, value=20)
    grip_strength_improvement = st.sidebar.slider("Βελτίωση δύναμης λαβής (%)", 0, 100, 50)
    dash_score_6months = st.sidebar.slider("DASH score στους 6 μήνες", 0, 100, 40)
    rom_extension_3m = st.sidebar.slider("ROM έκτασης (3μ)", 0, 180, 160)
    rom_flexion_3m = st.sidebar.slider("ROM κάμψης (3μ)", 0, 180, 150)
    rom_supination_3m = st.sidebar.slider("ROM υπτιασμού (3μ)", 0, 180, 140)
    rom_pronation_3m = st.sidebar.slider("ROM πρηνισμού (3μ)", 0, 180, 140)
    age_group = st.sidebar.selectbox("Ηλικιακή ομάδα", ["Νεαρός", "Μέσης ηλικίας", "Ηλικιωμένος"])
    risk_triad = st.sidebar.selectbox("Κίνδυνος (τριάδα)", ["Χαμηλός", "Μέτριος", "Υψηλός"])
    charlson_index = st.sidebar.slider("Δείκτης Charlson", 0, 10, 2)
    edmonton_frail_scale = st.sidebar.slider("Δείκτης Edmonton", 0, 10, 3)
    pase_score = st.sidebar.number_input("PASE Score", min_value=0, max_value=400, value=100)
    displacement = st.sidebar.selectbox("Μετατόπιση", ["Όχι", "Ναι"])
    fracture_stability = st.sidebar.selectbox("Σταθερότητα κατάγματος", ["Σταθερό", "Ασταθές"])
else:
    # Για απλό ασθενή βάζουμε προεπιλεγμένες ή απλές τιμές (μπορείς να τις αλλάξεις ή να τις αφήσεις κενές)
    osteoporosis = "Όχι"
    diabetes = "Όχι"
    fracture_type = "Απλό"
    physio_sessions = 20
    grip_strength_improvement = 50
    dash_score_6months = 40
    rom_extension_3m = 160
    rom_flexion_3m = 150
    rom_supination_3m = 140
    rom_pronation_3m = 140
    age_group = "Μέσης ηλικίας"
    risk_triad = "Μέτριος"
    charlson_index = 2
    edmonton_frail_scale = 3
    pase_score = 100
    displacement = "Όχι"
    fracture_stability = "Σταθερό"

# Μετατροπή σε αριθμητικά για το μοντέλο
input_dict = {
    "age": age,
    "sex": 0 if sex == "Άνδρας" else 1,
    "treatment_type": 0 if treatment_type == "Συντηρητική" else 1,
    "early_physiotherapy": 1 if early_physiotherapy == "Ναι" else 0,
    "osteoporosis": 1 if osteoporosis == "Ναι" else 0,
    "diabetes": 1 if diabetes == "Ναι" else 0,
    "fracture_type": 0 if fracture_type == "Απλό" else 1,
    "physio_sessions": physio_sessions,
    "grip_strength_improvement": grip_strength_improvement,
    "dash_score_6months": dash_score_6months,
    "rom_extension_3m": rom_extension_3m,
    "rom_flexion_3m": rom_flexion_3m,
    "rom_supination_3m": rom_supination_3m,
    "rom_pronation_3m": rom_pronation_3m,
    "age_group": {"Νεαρός": 0, "Μέσης ηλικίας": 1, "Ηλικιωμένος": 2}[age_group],
    "risk_triad": {"Χαμηλός": 0, "Μέτριος": 1, "Υψηλός": 2}[risk_triad],
    "charlson_index": charlson_index,
    "edmonton_frail_scale": edmonton_frail_scale,
    "pase_score": pase_score,
    "displacement": 1 if displacement == "Ναι" else 0,
    "fracture_stability": 0 if fracture_stability == "Σταθερό" else 1
}

input_df = pd.DataFrame([input_dict])
input_df = input_df[model_features]

if st.button("🔮 Υπολογισμός Χρόνου Αποκατάστασης"):
    prediction_weeks = model.predict(input_df)[0]
    st.subheader(f"🕒 Εκτιμώμενος Χρόνος Αποκατάστασης: **{prediction_weeks:.1f} εβδομάδες**")
st.info("Αυτή η πρόβλεψη βασίζεται σε εκπαιδευτικό μοντέλο με τεχνητά (συνθετικά) δεδομένα.")

avg_weeks = df["recovery_time_weeks"].mean()
st.markdown(f"📊 **Μέσος χρόνος αποκατάστασης στο δείγμα:** `{avg_weeks:.1f} εβδομάδες`")

    fig, ax = plt.subplots()
    sns.histplot(df["recovery_time_weeks"], kde=True, bins=20, ax=ax, color='skyblue')
    ax.axvline(prediction_weeks, color='red', linestyle='--', label='Η πρόβλεψή σας')
    ax.axvline(avg_weeks, color='green', linestyle='--', label='Μέσος όρος')
    ax.legend()
    st.pyplot(fig)

