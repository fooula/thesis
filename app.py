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

st.sidebar.info(
    "ℹ️ Για βοήθεια και επεξήγηση όρων, δείτε το παρακάτω πλαίσιο FAQ."
)
with st.sidebar.expander("❓ Συχνές Ερωτήσεις / Βοήθεια", expanded=True):
    st.markdown("""
**Charlson Comorbidity Index (CCI):**  
Ο CCI είναι ένας διεθνώς αναγνωρισμένος δείκτης που χρησιμοποιείται για την εκτίμηση της συνολικής βαρύτητας των συνοσηροτήτων ενός ασθενούς.
Κάθε χρόνια πάθηση (π.χ. διαβήτης, καρδιακή ανεπάρκεια, καρκίνος, ηπατική νόσος, κ.ά.) προσθέτει συγκεκριμένους βαθμούς στο συνολικό σκορ.
Όσο υψηλότερο το σκορ, τόσο μεγαλύτερος ο κίνδυνος για επιπλοκές, καθυστερημένη ανάρρωση.
Τιμές CCI: 0 (χωρίς συνοσηρότητες) έως 10+ (πολλαπλές ή σοβαρές συνοσηρότητες).

**Edmonton Frail Scale:**  
Κλίμακα ευαλωτότητας (0-17). Υψηλότερη τιμή σημαίνει μεγαλύτερη ευαλωτότητα/ευπάθεια.

**PASE Score:**  
Physical Activity Scale for the Elderly (0-400). Υψηλότερη τιμή σημαίνει περισσότερη φυσική δραστηριότητα.

**Displacement:**  
Μετατόπιση κατάγματος (0 = όχι, 1 = ναι).

**dash_score_6months:**  
Ερωτηματολόγιο DASH (Disabilities of the Arm, Shoulder and Hand) στους 6 μήνες (0-100, υψηλότερο = χειρότερη λειτουργικότητα).

**grip_strength_improvement:**  
Βελτίωση δύναμης λαβής (%) μετά τη θεραπεία.

**ROM:**  
Εύρος κίνησης καρπού στους 3 μήνες (μοίρες). Περιλαμβάνει:
- **rom_extension_3m:** Έκταση
- **rom_flexion_3m:** Κάμψη
- **rom_supination_3m:** Υπτιασμός
- **rom_pronation_3m:** Πρηνισμός

**fracture_stability:**  
Σταθερότητα κατάγματος (Σταθερό/Ασταθές)
    """)

# Mapping dictionaries για κατηγορικά
sex_map = {"Ανδρας": 0, "Γυναίκα": 1}
treatment_type_map = {"Χειρουργική": 1, "Μη Χειρουργική": 0}
early_physiotherapy_map = {"Όχι": 0, "Ναι": 1}
osteoporosis_map = {"Όχι": 0, "Ναι": 1}
diabetes_map = {"Όχι": 0, "Ναι": 1}
fracture_type_map = {"Εξωαρθρικό": 0, "Ενδοαρθρικό": 1, "Συντριπτικό": 2}
fracture_stability_map = {"Σταθερό": 0, "Ασταθές": 1}
age_group_map = {"<50": 0, "50-59": 1, "60-69": 2, "70-79": 3, "80+": 4}

st.title("Εκτίμηση Χρόνου Αποκατάστασης Μετά Από Κάταγμα Κερκίδας")

# Εισαγωγή τιμών από τον χρήστη
age = st.number_input("Ηλικία", min_value=18, max_value=100, value=60)
sex = st.selectbox("Φύλο", ["Ανδρας", "Γυναίκα"])
treatment_type = st.selectbox("Τύπος Θεραπείας", ["Χειρουργική", "Μη Χειρουργική"])
early_physiotherapy = st.selectbox("Έγκαιρη Φυσικοθεραπεία", ["Όχι", "Ναι"])
osteoporosis = st.selectbox("Οστεοπόρωση", ["Όχι", "Ναι"])
diabetes = st.selectbox("Διαβήτης", ["Όχι", "Ναι"])
fracture_type = st.selectbox("Τύπος Κατάγματος", ["Εξωαρθρικό", "Ενδοαρθρικό", "Συντριπτικό"])
physio_sessions = st.number_input("Συνεδρίες Φυσικοθεραπείας", min_value=0, max_value=30, value=10)
grip_strength_improvement = st.number_input("Βελτίωση Δύναμης Λαβής (%)", min_value=0.0, max_value=100.0, value=10.0)
dash_score_6months = st.number_input("DASH score στους 6 μήνες", min_value=0.0, max_value=100.0, value=20.0)
rom_extension_3m = st.number_input("ROM Extension 3 μήνες", min_value=0.0, max_value=180.0, value=60.0)
rom_flexion_3m = st.number_input("ROM Flexion 3 μήνες", min_value=0.0, max_value=180.0, value=60.0)
rom_supination_3m = st.number_input("ROM Supination 3 μήνες", min_value=0.0, max_value=180.0, value=60.0)
rom_pronation_3m = st.number_input("ROM Pronation 3 μήνες", min_value=0.0, max_value=180.0, value=60.0)
charlson_index = st.number_input("Charlson Comorbidity Index", min_value=0, max_value=10, value=2)
edmonton_frail_scale = st.number_input("Edmonton Frail Scale", min_value=0, max_value=17, value=5)
pase_score = st.number_input("PASE Score", min_value=0, max_value=400, value=100)
displacement = st.selectbox("Displacement", [0, 1])
fracture_stability = st.selectbox("Σταθερότητα Κατάγματος", ["Σταθερό", "Ασταθές"])

# Υπολογισμός risk_triad (Γυναίκα, ηλικία >65, οστεοπόρωση)
risk_triad = 1 if (sex == "Γυναίκα" and age > 65 and osteoporosis == "Ναι") else 0

# Υπολογισμός age_group με βάση την ηλικία
if age < 50:
    age_group = "<50"
elif 50 <= age < 60:
    age_group = "50-59"
elif 60 <= age < 70:
    age_group = "60-69"
elif 70 <= age < 80:
    age_group = "70-79"
else:
    age_group = "80+"

# Δημιουργία input DataFrame με mapping
input_dict = {
    "age": age,
    "sex": sex_map[sex],
    "treatment_type": treatment_type_map[treatment_type],
    "early_physiotherapy": early_physiotherapy_map[early_physiotherapy],
    "osteoporosis": osteoporosis_map[osteoporosis],
    "diabetes": diabetes_map[diabetes],
    "fracture_type": fracture_type_map[fracture_type],
    "physio_sessions": physio_sessions,
    "grip_strength_improvement": grip_strength_improvement,
    "dash_score_6months": dash_score_6months,
    "rom_extension_3m": rom_extension_3m,
    "rom_flexion_3m": rom_flexion_3m,
    "rom_supination_3m": rom_supination_3m,
    "rom_pronation_3m": rom_pronation_3m,
    "age_group": age_group_map[age_group],
    "risk_triad": risk_triad,
    "charlson_index": charlson_index,
    "edmonton_frail_scale": edmonton_frail_scale,
    "pase_score": pase_score,
    "displacement": displacement,
    "fracture_stability": fracture_stability_map[fracture_stability],
}
input_df = pd.DataFrame([input_dict])

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



