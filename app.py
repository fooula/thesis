import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Φόρτωση εκπαιδευμένου μοντέλου
model = joblib.load("xgboost_model.pkl")

# Φόρτωση dataset για συγκρίσεις
try:
    data = pd.read_csv("distal_radius_recovery.csv")
except FileNotFoundError:
    st.error("Το αρχείο dataset.csv δεν βρέθηκε. Παρακαλώ ανέβασε το dataset.")
    st.stop()

st.title("Εκτίμηση Χρόνου Αποκατάστασης Μετά από Κάταγμα Κερκίδας")

st.header("🔍 Εισαγωγή Χαρακτηριστικών Ασθενούς")

# Εισαγωγή μεταβλητών από τον χρήστη
age = st.number_input("Ηλικία", min_value=0, max_value=120, value=50)
sex = st.selectbox("Φύλο", ["Άνδρας", "Γυναίκα"])
treatment_type = st.selectbox("Τύπος Θεραπείας", ["Χειρουργείο", "Συντηρητική"])
early_physiotherapy = st.selectbox("Έναρξη Φυσικοθεραπείας σε <1 εβδομάδα;", ["Όχι", "Ναι"])
osteoporosis = st.selectbox("Οστεοπόρωση", ["Όχι", "Ναι"])
diabetes = st.selectbox("Διαβήτης", ["Όχι", "Ναι"])
fracture_type = st.selectbox("Τύπος Κατάγματος", ["Ενδοαρθρικό", "Εκτός Αρθρικής Επιφάνειας"])
physio_sessions = st.number_input("Αριθμός Φυσικοθεραπειών", min_value=0, value=10)
grip_strength_improvement = st.number_input("Βελτίωση Δύναμης Λαβής (%)", min_value=0.0, max_value=100.0, value=30.0)
dash_score_6months = st.number_input("DASH Score στους 6 μήνες", min_value=0.0, max_value=100.0, value=20.0)
rom_extension_3m = st.number_input("ROM Extension στους 3 μήνες", value=0.0)
rom_flexion_3m = st.number_input("ROM Flexion στους 3 μήνες", value=0.0)
rom_supination_3m = st.number_input("ROM Supination στους 3 μήνες", value=0.0)
rom_pronation_3m = st.number_input("ROM Pronation στους 3 μήνες", value=0.0)
age_group = st.selectbox("Ηλικιακή Ομάδα", ["<40", "40-60", ">60"])
risk_triad = st.selectbox("Τριάδα Κινδύνου", ["Χαμηλός", "Μέτριος", "Υψηλός"])
charlson_index = st.number_input("Δείκτης Συννοσηρότητας Charlson", value=0)
edmonton_frail_scale = st.number_input("Κλίμακα Ευπάθειας Edmonton", value=0)
pase_score = st.number_input("PASE Score (φυσική δραστηριότητα)", value=0)
displacement = st.selectbox("Μετατόπιση Κατάγματος", ["Όχι", "Ναι"])
fracture_stability = st.selectbox("Σταθερότητα Κατάγματος", ["Σταθερό", "Ασταθές"])

# Μετατροπή εισόδου σε μορφή DataFrame
input_dict = {
    "age": age,
    "sex": sex,
    "treatment_type": treatment_type,
    "early_physiotherapy": early_physiotherapy,
    "osteoporosis": osteoporosis,
    "diabetes": diabetes,
    "fracture_type": fracture_type,
    "physio_sessions": physio_sessions,
    "grip_strength_improvement": grip_strength_improvement,
    "dash_score_6months": dash_score_6months,
    "rom_extension_3m": rom_extension_3m,
    "rom_flexion_3m": rom_flexion_3m,
    "rom_supination_3m": rom_supination_3m,
    "rom_pronation_3m": rom_pronation_3m,
    "age_group": age_group,
    "risk_triad": risk_triad,
    "charlson_index": charlson_index,
    "edmonton_frail_scale": edmonton_frail_scale,
    "pase_score": pase_score,
    "displacement": displacement,
    "fracture_stability": fracture_stability,
}

input_df = pd.DataFrame([input_dict])

# Κωδικοποίηση μεταβλητών όπως στο dataset
categorical_cols = ['sex', 'treatment_type', 'early_physiotherapy', 'osteoporosis', 'diabetes', 'fracture_type', 'age_group', 'risk_triad', 'displacement', 'fracture_stability']

for col in categorical_cols:
    le = LabelEncoder()
    le.fit(data[col])
    try:
        input_df[col] = le.transform(input_df[col])
    except ValueError:
        st.warning(f"Η τιμή στη στήλη {col} δεν υπήρχε στο dataset και αγνοήθηκε.")
        input_df[col] = -1  # τιμή placeholder για άγνωστες κατηγορίες

# Επιλογή χαρακτηριστικών όπως στο training
model_features = data.drop(columns=['recovery_time_weeks']).columns.tolist()

if 'patient_id' in model_features:
    model_features.remove('patient_id')
input_df = input_df[model_features]

# Πρόβλεψη
prediction_weeks = model.predict(input_df)[0]
st.subheader(f"📅 Εκτιμώμενος χρόνος αποκατάστασης: **{prediction_weeks:.1f} εβδομάδες**")

# Σύγκριση με παρόμοιους ασθενείς
st.header("📊 Σύγκριση με παρόμοιους ασθενείς")
compare_features = ['age_group', 'sex', 'treatment_type', 'fracture_type']

# Φιλτράρισμα παρόμοιων
similar_patients = data.copy()
for feat in compare_features:
    if feat in categorical_cols:
        le = LabelEncoder()
        le.fit(data[feat])
        
        if input_df[feat].iloc[0] in le.classes_:
           val = le.transform(input_df[feat])[0]
        else:
         # Διάλεξε μια στρατηγική
            val = -1  # ή μήνυμα λάθους / default τιμή / skip

        similar_patients = similar_patients[le.transform(similar_patients[feat]) == val]
    else:
        similar_patients = similar_patients[similar_patients[feat] == input_df[feat].values[0]]

if len(similar_patients) >= 5:
    mean_weeks = similar_patients['recovery_time'].mean()
    st.write(f"Μέσος χρόνος αποκατάστασης για παρόμοιους ασθενείς: **{mean_weeks:.1f} εβδομάδες**")

    fig, ax = plt.subplots()
    sns.histplot(similar_patients['recovery_time'], bins=20, kde=True, ax=ax)
    ax.axvline(prediction_weeks, color='red', linestyle='--', label='Η πρόβλεψή σας')
    ax.set_xlabel('Χρόνος αποκατάστασης (εβδομάδες)')
    ax.set_ylabel('Αριθμός ασθενών')
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Δεν βρέθηκαν αρκετοί παρόμοιοι ασθενείς. Δοκίμασε με πιο γενικά χαρακτηριστικά ή μεγαλύτερο dataset.")
