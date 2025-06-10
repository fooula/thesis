import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go


# Φόρτωση εκπαιδευμένου μοντέλου
model = joblib.load("xgboost_model.pkl")

st.set_page_config(page_title="Εκτίμηση Αποκατάστασης", layout="centered")
st.title("🦴 Εκτίμηση Χρόνου Αποκατάστασης Κατάγματος Κερκίδας")

df = pd.read_csv("distal_radius_recovery_rom_included.csv")
X = df.drop(columns=["recovery_time_weeks"])
y = df["recovery_time_weeks"]

# Είσοδοι Χρήστη
age = st.number_input("Ηλικία", min_value=0, max_value=100, value=50)
sex = st.selectbox("Φύλο", ["Άνδρας", "Γυναίκα"])
treatment_type = st.selectbox("Τύπος Θεραπείας", ["Συντηρητική", "Χειρουργική"])
early_physiotherapy = st.selectbox("Έγκαιρη Φυσικοθεραπεία;", ["Όχι", "Ναι"])
osteoporosis = st.selectbox("Οστεοπόρωση;", ["Όχι", "Ναι"])
diabetes = st.selectbox("Διαβήτης;", ["Όχι", "Ναι"])
fracture_type = st.selectbox("Τύπος Κατάγματος", ["Απλό", "Σύνθετο", "Ενδοαρθρικό"])
physio_sessions = st.number_input("Αριθμός Φυσικοθεραπειών", min_value=0, value=10)
grip_strength_improvement = st.number_input("Βελτίωση Δύναμης Λαβής (%)", min_value=0.0, max_value=100.0, value=30.0)
dash_score_6months = st.number_input("DASH Score (6 μήνες)", min_value=0.0, max_value=100.0, value=20.0)

rom_extension_3m = st.number_input("ROM - Έκταση (3 μήνες)", min_value=0.0, value=45.0)
rom_flexion_3m = st.number_input("ROM - Κάμψη (3 μήνες)", min_value=0.0, value=50.0)
rom_supination_3m = st.number_input("ROM - Υπτιασμός (3 μήνες)", min_value=0.0, value=60.0)
rom_pronation_3m = st.number_input("ROM - Πρηνισμός (3 μήνες)", min_value=0.0, value=55.0)

# === Κωδικοποίηση Κατηγορικών Μεταβλητών ===
sex_map = {"Άνδρας": 0, "Γυναίκα": 1}
treatment_map = {"Συντηρητική": 0, "Χειρουργική": 1}
physio_map = {"Όχι": 0, "Ναι": 1}
fracture_map = {"Απλό": 0, "Σύνθετο": 1, "Ενδοαρθρικό": 2}

# === Δημιουργία DataFrame με τις στήλες που έχουμε ===
input_data = pd.DataFrame([[
    age,
    sex_map[sex],
    treatment_map[treatment_type],
    physio_map[early_physiotherapy],
    physio_map[osteoporosis],
    physio_map[diabetes],
    fracture_map[fracture_type],
    physio_sessions,
    grip_strength_improvement,
    dash_score_6months,
    rom_extension_3m,
    rom_flexion_3m,
    rom_supination_3m,
    rom_pronation_3m
]], columns=[
    'age',
    'sex',
    'treatment_type',
    'early_physiotherapy',
    'osteoporosis',
    'diabetes',
    'fracture_type',
    'physio_sessions',
    'grip_strength_improvement',
    'dash_score_6months',
    'rom_extension_3m',
    'rom_flexion_3m',
    'rom_supination_3m',
    'rom_pronation_3m'
])

# === Συμπλήρωση υπολοίπων στηλών που περίμενε το μοντέλο ===
model_features = model.get_booster().feature_names
for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0  # Μηδενική τιμή για στήλες που αφαιρέθηκαν

# === Αναδιάταξη των στηλών σύμφωνα με τη σειρά του μοντέλου ===
input_data = input_data[model_features]

# === Πρόβλεψη ===
if st.button("🔮 Υπολογισμός Χρόνου Αποκατάστασης"):
    prediction = model.predict(input_data)[0]
    st.success(f"✅ Εκτιμώμενος χρόνος αποκατάστασης: **{prediction:.2f} εβδομάδες**")

mean_recovery = y.mean()
st.write(f"Μέσος Χρόνος Αποκατάστασης στο Dataset: {mean_recovery:.2f} εβδομάδες")


# Επιλογή χαρακτηριστικών που θέλεις να συγκρίνεις
radar_features = [
    'age',
    'physio_sessions',
    'grip_strength_improvement',
    'dash_score_6months',
    'rom_extension_3m',
    'rom_flexion_3m',
    'rom_supination_3m',
    'rom_pronation_3m'
]


# Υπολογισμός μέσου όρου dataset
average_values = df[radar_features].mean().tolist()

# Ανάκτηση των τιμών του τρέχοντος ασθενούς
patient_values = [input_data[feature] for feature in radar_features]

# Δημιουργία radar chart
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=average_values,
    theta=radar_features,
    fill='toself',
    name='Μέσος Όρος'
))

fig.add_trace(go.Scatterpolar(
    r=patient_values,
    theta=radar_features,
    fill='toself',
    name='Ασθενής'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True),
    ),
    showlegend=True,
    title='Σύγκριση Ασθενούς με Μέσο Όρο'
)

st.plotly_chart(fig)

st.markdown("## 🧪 What-if Ανάλυση")
st.markdown("Πειραματίσου με διαφορετικές τιμές για βασικές παραμέτρους και δες πώς επηρεάζεται η πρόβλεψη του μοντέλου.")

# Τιμές για επιλογές
sex_options = df['sex'].unique().tolist()
fracture_options = df['fracture_type'].unique().tolist()
treatment_options = df['treatment_type'].unique().tolist()

# Sidebar ή κύριο UI
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Ηλικία", min_value=20, max_value=90, value=65)
    grip_strength_improvement = st.slider("Βελτίωση δύναμης λαβής (%)", min_value=0, max_value=100, value=50)
    physio_sessions = st.slider("Συνεδρίες φυσικοθεραπείας", min_value=0, max_value=50, value=20)

with col2:
    sex = st.selectbox("Φύλο", sex_options)
    fracture_type = st.selectbox("Τύπος κατάγματος", fracture_options)
    treatment_type = st.selectbox("Τύπος θεραπείας", treatment_options)
    osteoporosis = st.radio("Οστεοπόρωση", [0, 1], format_func=lambda x: "Ναι" if x == 1 else "Όχι")
    diabetes = st.radio("Διαβήτης", [0, 1], format_func=lambda x: "Ναι" if x == 1 else "Όχι")
    early_physiotherapy = st.radio("Έγκαιρη φυσικοθεραπεία", [0, 1], format_func=lambda x: "Ναι" if x == 1 else "Όχι")

# Δημιουργία input DataFrame
input_data = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "treatment_type": treatment_type,
    "early_physiotherapy": early_physiotherapy,
    "osteoporosis": osteoporosis,
    "diabetes": diabetes,
    "fracture_type": fracture_type,
    "physio_sessions": physio_sessions,
    "grip_strength_improvement": grip_strength_improvement,
}])

# Προετοιμασία input_data ώστε να ταιριάζει με τις στήλες του μοντέλου
input_data["sex"] = input_data["sex"].map(sex_map)
input_data["treatment_type"] = input_data["treatment_type"].map(treatment_map)
input_data["fracture_type"] = input_data["fracture_type"].map(fracture_map)

# Προσθήκη μηδενικών σε features που λείπουν
for col in model.get_booster().feature_names:
    if col not in input_data.columns:
        input_data[col] = 0

# Αναδιάταξη στη σωστή σειρά
input_data = input_data[model.get_booster().feature_names]

# Πρόβλεψη
predicted_weeks = model.predict(input_data)[0]


# Εμφάνιση πρόβλεψης
st.success(f"📅 Εκτιμώμενος χρόνος αποκατάστασης: **{predicted_weeks:.1f} εβδομάδες**")
