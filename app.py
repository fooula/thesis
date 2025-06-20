import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Φόρτωση μοντέλου και δεδομένων
model = joblib.load("xgboost_model.pkl")
df = pd.read_csv("distal_radius_recovery.csv")

# Ορισμός των features που περιμένει το μοντέλο
model_features = [
    "age", "sex", "treatment_type", "early_physiotherapy", "osteoporosis", "diabetes",
    "fracture_type", "physio_sessions", "grip_strength_improvement", "dash_score_6months",
    "rom_extension_3m", "rom_flexion_3m", "rom_supination_3m", "rom_pronation_3m",
    "age_group", "risk_triad", "charlson_index", "edmonton_frail_scale", "pase_score",
    "displacement", "fracture_stability"
]

# Mapping dictionaries για κατηγορικά
sex_map = {"male": 0, "female": 1}
treatment_type_map = {"operative": 1, "nonoperative": 0}
fracture_type_map = {"A": 0, "B": 1, "C": 2}
age_group_map = {"<50": 0, "50-59": 1, "60-69": 2, "70-79": 3, "80+": 4}
fracture_stability_map = {"stable": 0, "unstable": 1}

# Sidebar FAQ
with st.sidebar.expander("ℹ️ Τι σημαίνουν οι όροι;"):
    st.markdown("""
- **Charlson Comorbidity Index (CCI)**: 
    - Ο CCI είναι ένας διεθνώς αναγνωρισμένος δείκτης που χρησιμοποιείται για την εκτίμηση της συνολικής βαρύτητας των συνοσηροτήτων ενός ασθενούς.
    - Κάθε χρόνια πάθηση (π.χ. διαβήτης, καρδιακή ανεπάρκεια, καρκίνος, ηπατική νόσος, κ.ά.) προσθέτει συγκεκριμένους βαθμούς στο συνολικό σκορ.
    - Όσο υψηλότερο το σκορ, τόσο μεγαλύτερος ο κίνδυνος για επιπλοκές, καθυστερημένη ανάρρωση.
    - Τιμές CCI: 0 (χωρίς συνοσηρότητες) έως 10+ (πολλαπλές ή σοβαρές συνοσηρότητες).
- **Edmonton Frail Scale**: Κλίμακα ευαλωτότητας (0-17). Υψηλότερη τιμή σημαίνει μεγαλύτερη ευαλωτότητα/ευπάθεια.
- **PASE Score**: Physical Activity Scale for the Elderly (0-400). Υψηλότερη τιμή σημαίνει περισσότερη φυσική δραστηριότητα.
- **Displacement**: Μετατόπιση κατάγματος (0 = όχι, 1 = ναι).
- **risk_triad**: Συνδυαστικός δείκτης κινδύνου (Γυναίκες >65 ετών με οστεοπόρωση) (0 = όχι, 1 = ναι).
- **dash_score_6months**: Ερωτηματολόγιο DASH (Disabilities of the Arm, Shoulder and Hand) στους 6 μήνες (0-100, υψηλότερο = χειρότερη λειτουργικότητα).
- **grip_strength_improvement**: Βελτίωση δύναμης λαβής (%) μετά τη θεραπεία.
- **ROM**: Εύρος κίνησης καρπού στους 3 μήνες (μοίρες). Περιλαμβάνει:
    - **rom_extension_3m**: Έκταση
    - **rom_flexion_3m**: Κάμψη
    - **rom_supination_3m**: Υπτιασμός
    - **rom_pronation_3m**: Πρηνισμός
    """)

st.title("Εκτίμηση Χρόνου Αποκατάστασης Μετά Από Κάταγμα Κερκίδας")

# Εισαγωγή τιμών από τον χρήστη
age = st.number_input("Ηλικία", min_value=18, max_value=100, value=60)
sex = st.selectbox("Φύλο", ["Ανδρας", "Γυναίκα"])
treatment_type = st.selectbox("Τύπος Θεραπείας", ["Χειρουργική", "Μη Χειρουργική"])
early_physiotherapy = st.selectbox("Έγκαιρη Φυσικοθεραπεία", [0, 1])
osteoporosis = st.selectbox("Οστεοπόρωση", [0, 1])
diabetes = st.selectbox("Διαβήτης", [0, 1])
fracture_type = st.selectbox("Τύπος Κατάγματος", ["A", "B", "C"])
physio_sessions = st.number_input("Συνεδρίες Φυσικοθεραπείας", min_value=0, max_value=30, value=10)
grip_strength_improvement = st.number_input("Βελτίωση Δύναμης Λαβής (%)", min_value=0.0, max_value=100.0, value=10.0)
dash_score_6months = st.number_input("DASH score στους 6 μήνες", min_value=0.0, max_value=100.0, value=20.0)
rom_extension_3m = st.number_input("ROM Extension 3 μήνες", min_value=0.0, max_value=180.0, value=60.0)
rom_flexion_3m = st.number_input("ROM Flexion 3 μήνες", min_value=0.0, max_value=180.0, value=60.0)
rom_supination_3m = st.number_input("ROM Supination 3 μήνες", min_value=0.0, max_value=180.0, value=60.0)
rom_pronation_3m = st.number_input("ROM Pronation 3 μήνες", min_value=0.0, max_value=180.0, value=60.0)
age_group = st.selectbox("Ηλικιακή Ομάδα", ["<50", "50-59", "60-69", "70-79", "80+"])
risk_triad = st.selectbox("Risk Triad", [0, 1])
charlson_index = st.number_input("Charlson Comorbidity Index", min_value=0, max_value=10, value=2)
edmonton_frail_scale = st.number_input("Edmonton Frail Scale", min_value=0, max_value=17, value=5)
pase_score = st.number_input("PASE Score", min_value=0, max_value=400, value=100)
displacement = st.selectbox("Displacement", [0, 1])
fracture_stability = st.selectbox("Σταθερότητα Κατάγματος", ["stable", "unstable"])

# Δημιουργία input DataFrame
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

# Mapping κατηγορικών
input_df["sex"] = input_df["sex"].map(sex_map)
input_df["treatment_type"] = input_df["treatment_type"].map(treatment_type_map)
input_df["fracture_type"] = input_df["fracture_type"].map(fracture_type_map)
input_df["age_group"] = input_df["age_group"].map(age_group_map)
input_df["fracture_stability"] = input_df["fracture_stability"].map(fracture_stability_map)

# Έλεγχος για NaN μετά το mapping
if input_df[model_features].isnull().any().any():
    st.error("Κάποια πεδία δεν έχουν σωστή τιμή. Ελέγξτε τα κατηγορικά πεδία.")
    st.stop()

# Τελική σειρά στηλών
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

