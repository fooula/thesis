import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost



# Φόρτωση μοντέλου και δεδομένων
model = joblib.load("xgboost_model.pkl")
df = pd.read_csv("distal_radius_fracture_recovery.csv")

# Ορισμός των features που περιμένει το μοντέλο
model_features = [
    "age", "sex", "dominant_hand_injured", "osteoporosis", "charlson_index",
    "edmonton_frail_scale", "pase_score", "risk_triad", "social_support",
    "fracture_type", "displacement", "fracture_stability", "operative_treatment",
    "immobilization_days"
]




with st.sidebar.expander("❓ Συχνές Ερωτήσεις / Βοήθεια"):
    st.markdown("""
- **Charlson Comorbidity Index (CCI):**
    - Ο CCI είναι ένας διεθνώς αναγνωρισμένος δείκτης που χρησιμοποιείται για την εκτίμηση της συνολικής βαρύτητας των παραλληλων παθησεων ενός ασθενούς.
    - Κάθε χρόνια πάθηση (π.χ. διαβήτης, καρδιακή ανεπάρκεια, καρκίνος, ηπατική νόσος, ΧΑΠ, άνοια, κ.ά.) προσθέτει συγκεκριμένους βαθμούς στο συνολικό σκορ, ανάλογα με τη βαρύτητα και τον αριθμό των συνοσηροτήτων.
    - Όσο υψηλότερο το σκορ, τόσο μεγαλύτερος ο κίνδυνος για επιπλοκές, καθυστερημένη ανάρρωση και θνητότητα.
    - Τιμές CCI: 0 (χωρίς συνοσηρότητες) έως 10+ (πολλαπλές ή σοβαρές συνοσηρότητες).
    - Χρησιμοποιείται ευρέως για την πρόβλεψη της πρόγνωσης σε διάφορες ιατρικές καταστάσεις.
    - [Charlson ME, et al. J Chronic Dis. 1987](https://pubmed.ncbi.nlm.nih.gov/3558716/)

- **Edmonton Frail Scale:**
    - Κλίμακα ευπαθειας (0-17) που αξιολογεί πολλαπλές διαστάσεις (γνωστική λειτουργία, γενική υγεία, ανεξαρτησία, κοινωνική υποστήριξη, διατροφή, φαρμακευτική αγωγή, ισορροπία/λειτουργικότητα).
    - Υψηλότερη τιμή σημαίνει μεγαλύτερη ευαλωτότητα/ευπάθεια και αυξημένο κίνδυνο για επιπλοκές, πτώσεις και δυσμενή έκβαση.
    - Χρησιμοποιείται κυρίως σε ηλικιωμένους για την εκτίμηση της συνολικής κατάστασης υγείας.
    - [Rolfson DB, et al. Age Ageing. 2006](https://pubmed.ncbi.nlm.nih.gov/16641176/)

- **PASE Score:**
    - Physical Activity Scale for the Elderly (0-400). Κλίμακα που μετρά τη φυσική δραστηριότητα σε ηλικιωμένους, λαμβάνοντας υπόψη εργασία, άσκηση και καθημερινές δραστηριότητες.
    - Υψηλότερη τιμή σημαίνει περισσότερη φυσική δραστηριότητα και καλύτερη λειτουργικότητα.
    - Χρησιμοποιείται για την αξιολόγηση της φυσικής κατάστασης και την πρόβλεψη της αποκατάστασης.
    - [Washburn RA, et al. J Clin Epidemiol. 1993](https://pubmed.ncbi.nlm.nih.gov/8410095/)

- **Displacement:**
    - Μετατόπιση κατάγματος (0 = όχι, 1 = ναι). Η παρουσία μετατόπισης σχετίζεται με μεγαλύτερη βαρύτητα κάκωσης και συχνά απαιτεί πιο επιθετική θεραπεία.
    - [Müller ME, et al. AO classification of fractures. Injury. 1990](https://pubmed.ncbi.nlm.nih.gov/2221851/)

- **DASH Score:**
    - Ερωτηματολόγιο DASH (Disabilities of the Arm, Shoulder and Hand) στους 6 μήνες (0-100, υψηλότερο = χειρότερη λειτουργικότητα).
    - Αξιολογεί τη λειτουργικότητα του άνω άκρου μετά από τραυματισμούς ή παθήσεις.
    - [Hudak PL, et al. The DASH outcome measure. Am J Ind Med. 1996](https://pubmed.ncbi.nlm.nih.gov/8773720/)

- **Βελτίωση Δύναμης Λαβής:**  
Ποσοστιαία βελτίωση της δύναμης λαβής (%) μετά τη θεραπεία σε σχέση με την αρχική μέτρηση.

- **ROM (Range of Motion):**
    - Εύρος κίνησης καρπού στους 3 μήνες (μοίρες). Αξιολογείται σε τέσσερις βασικές κινήσεις:
        - **rom_extension_3m:** Έκταση (extension)
        - **rom_flexion_3m:** Κάμψη (flexion)
        - **rom_supination_3m:** Υπτιασμός (supination)
        - **rom_pronation_3m:** Πρηνισμός (pronation)
    - Η αποκατάσταση του ROM είναι κρίσιμη για την επάνοδο στη λειτουργικότητα.
    - [MacDermid JC, et al. Measurement of wrist motion. J Hand Ther. 2001](https://pubmed.ncbi.nlm.nih.gov/11511019/)

- **Σταθερότητα Κατάγματος:**  
Σταθερότητα κατάγματος (Σταθερό/Ασταθές). Η σταθερότητα καθορίζει τη θεραπευτική προσέγγιση και την πρόγνωση.
    """)

# Mapping dictionaries για κατηγορικά
sex_map = {"Ανδρας": 0, "Γυναίκα": 1}
dominant_hand_injured_map = {"Όχι": 0, "Ναι": 1}
osteoporosis_map = {"Όχι": 0, "Ναι": 1}
risk_triad_map = {"Όχι": 0, "Ναι": 1}
social_support_map = {"Καμία : Ζω μόνος/η και δεν υπάρχει διαθέσιμο άτομο να με βοηθά τακτικά.": 0, "Μερική : Έχω περιστασιακή βοήθεια από συγγενή, φίλο ή γείτονα.": 1, "Σταθερή : Eχω σταθερή και ενεργή υποστήριξη από οικογένεια ή φροντιστή": 2}
fracture_type_map = {"Εξωαρθρικό": 0, "Ενδοαρθρικό": 1}
displacement_map = {"Όχι": 0, "Ναι": 1}
fracture_stability_map = {"Σταθερό": 0, "Ασταθές": 1}
operative_treatment_map = {"Όχι": 0, "Ναι": 1}

st.title("Εκτίμηση Χρόνου Αποκατάστασης Μετά Από Κάταγμα Κερκίδας")

# Εισαγωγή τιμών από τον χρήστη
age = st.number_input("Ηλικία", min_value=18, max_value=100, value=60)
sex = st.selectbox("Φύλο", ["Ανδρας", "Γυναίκα"])
dominant_hand_injured = st.selectbox("Τραυματισμός κυρίαρχου άνω άκρου;", ["Όχι", "Ναι"])
osteoporosis = st.selectbox("Οστεοπόρωση", ["Όχι", "Ναι"])
charlson_index = st.number_input("Charlson Comorbidity Index (Βοηθεια Υπολογισμου: https://unaettie.com/en-us/pz/charlson.php?utm_source=chatgpt.com)", min_value=0, max_value=10, value=2)
edmonton_frail_scale = st.number_input("Edmonton Frail Scale (Βοηθεια Υπολογισμου: https://qxmd.com/calculate/calculator_595/edmonton-frail-scale?utm_source=chatgpt.com)", min_value=0, max_value=17, value=5)
pase_score = st.number_input("PASE Score", min_value=0, max_value=400, value=100)
social_support = st.selectbox("Κοινωνική/Οικογενειακή Υποστήριξη", ["Καμία : Ζω μόνος/η και δεν υπάρχει διαθέσιμο άτομο να με βοηθά τακτικά.", "Μερική : Έχω περιστασιακή βοήθεια από συγγενή, φίλο ή γείτονα.", "Σταθερή : Eχω σταθερή και ενεργή υποστήριξη από οικογένεια ή φροντιστή"])
fracture_type = st.selectbox("Τύπος Κατάγματος", ["Εξωαρθρικό", "Ενδοαρθρικό"])
displacement = st.selectbox("Παρεκτόπιση", ["Όχι", "Ναι"])
fracture_stability = st.selectbox("Σταθερότητα Κατάγματος", ["Σταθερό", "Ασταθές"])
operative_treatment = st.selectbox("Χειρουργική Αντιμετώπιση", ["Όχι", "Ναι"])
immobilization_days = st.number_input("Διάρκεια Ακινητοποίησης (ημέρες)", min_value=10, max_value=60, value=30)

# Υπολογισμός risk_triad αυτόματα (χωρίς selectbox)
risk_triad = 1 if (sex == "Γυναίκα" and age > 65 and osteoporosis == "Ναι") else 0

# Δημιουργία input DataFrame με mapping
input_dict = {
    "age": age,
    "sex": sex_map[sex],
    "dominant_hand_injured": dominant_hand_injured_map[dominant_hand_injured],
    "osteoporosis": osteoporosis_map[osteoporosis],
    "charlson_index": charlson_index,
    "edmonton_frail_scale": edmonton_frail_scale,
    "pase_score": pase_score,
    # χρησιμοποιούμε απευθείας την integer τιμή που υπολογίσαμε
    "risk_triad": risk_triad,
    "social_support": social_support_map[social_support],
    "fracture_type": fracture_type_map[fracture_type],
    "displacement": displacement_map[displacement],
    "fracture_stability": fracture_stability_map[fracture_stability],
    "operative_treatment": operative_treatment_map[operative_treatment],
    "immobilization_days": immobilization_days,
}

input_df = pd.DataFrame([input_dict])
input_df = input_df[model_features]

# Έλεγχος για NaN μετά το mapping
if input_df[model_features].isnull().any().any():
    st.error("Κάποια πεδία δεν έχουν σωστή τιμή. Ελέγξτε τα κατηγορικά πεδία.")
    st.stop()

# Τελική σειρά στηλών
input_df = input_df[model_features]

# --- Ενημερωμένος υπολογισμός Risk Score με βάρη από SHAP feature‐importance ---
risk_score = (
    # κανονικοποίηση συνεχειών: immobilization_days 10–60 → 0–1
    (immobilization_days - 10) / (60 - 10) * 0.144 +

    # binary / ordinal features
    risk_triad * 0.137 +
    (age / 100) * 0.124 +
    operative_treatment_map[operative_treatment] * 0.111 +
    osteoporosis_map[osteoporosis] * 0.105 +
    fracture_type_map[fracture_type] * 0.073 +
    fracture_stability_map[fracture_stability] * 0.068 +

    # κλίμακες: charlson_index 0–10, edmonton 0–17, pase 0–400
    (charlson_index / 10) * 0.059 +
    (pase_score / 400) * 0.044 +
    (edmonton_frail_scale / 17) * 0.043 +

    displacement_map[displacement] * 0.041 +
    social_support_map[social_support] * 0.035 +
    sex_map[sex] * 0.009
)

st.metric("Risk Score", f"{risk_score:.2f}")

# Κατηγοριοποίηση Risk Score
if risk_score < 0.25:
    risk_level = "🟢 Χαμηλός Κίνδυνος"
    risk_color = "green"
elif risk_score < 0.45:
    risk_level = "🟡 Μέτριος Κίνδυνος"
    risk_color = "orange"
else:
    risk_level = "🔴 Υψηλός Κίνδυνος"
    risk_color = "red"

st.markdown(
    f"<span style='color:{risk_color}; font-weight:bold; font-size:1.2em;'>Εκτίμηση: {risk_level}</span>",
    unsafe_allow_html=True
)

with st.sidebar.expander("ℹ️ Τι είναι το Risk Score;"):
    st.markdown("""
**Risk Score**: Ένας σύνθετος δείκτης που συνδυάζει τα σημαντικότερα κλινικά και λειτουργικά χαρακτηριστικά (ημερες ακινητοποιησης, οστεοπόρωση, risk triad, παραλληλες παθησεις, σταθερότητα κατάγματος, ηλικία, τύπος θεραπείας, τύπος κατάγματος, μετατόπιση, ευπαθεια, φυσική δραστηριότητα, λειτουργικότητα, κοινωνική/οικογενειακή Υποστήριξη).
Όσο υψηλότερος ο δείκτης, τόσο μεγαλύτερη η προβλεπόμενη δυσκολία αποκατάστασης.
Ο δείκτης αυτός είναι ενδεικτικός και δεν αντικαθιστά την ιατρική κρίση.
    """)

if st.button("🔮 Υπολογισμός Χρόνου Αποκατάστασης"):
    prediction_weeks = model.predict(input_df)[0]
    st.subheader(f"🕒 Εκτιμώμενος Χρόνος Αποκατάστασης: **{prediction_weeks:.1f} εβδομάδες**")
    st.info("⚠️ Αυτή η πρόβλεψη βασίζεται σε εκπαιδευτικό μοντέλο με τεχνητά (συνθετικά) δεδομένα. ⚠️")

    avg_weeks = df["recovery_time_weeks"].mean()
    st.markdown(f"📊 **Μέσος χρόνος αποκατάστασης στο δείγμα:** `{avg_weeks:.1f} εβδομάδες`")

    fig, ax = plt.subplots()
    sns.histplot(df["recovery_time_weeks"], kde=True, bins=20, ax=ax, color='skyblue')
    ax.axvline(prediction_weeks, color='red', linestyle='--', label='Η πρόβλεψή σας')
    ax.axvline(avg_weeks, color='green', linestyle='--', label='Μέσος όρος')
    ax.legend()
    st.pyplot(fig)







