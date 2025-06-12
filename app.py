import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go


# Φόρτωση εκπαιδευμένου μοντέλου
model = joblib.load("xgboost_model.pkl")

st.set_page_config(page_title="Εκτίμηση Αποκατάστασης", layout="centered")
st.title("🦴 Εκτίμηση Χρόνου Αποκατάστασης Κατάγματος Κερκίδας")

df = pd.read_csv("distal_radius_recovery.csv")
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
    st.session_state['initial_prediction'] = prediction
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


st.markdown("---")
st.subheader("🧪 What-if Ανάλυση")

st.write("Πειραματίσου με διαφορετικές τιμές σε βασικές παραμέτρους για να δεις πώς επηρεάζεται ο χρόνος αποκατάστασης.")

# Επιλογή τιμών για την what-if ανάλυση
what_if_physio_sessions = st.slider("Τι θα γινοταν αν ο ασθενής έκανε περισσότερες φυσικοθεραπείες;", 
                                     min_value=0, max_value=50, value=int(physio_sessions))

what_if_grip_strength = st.slider("Τι θα γινοταν αν υπήρχε μεγαλύτερη βελτίωση δύναμης λαβής (%);", 
                                   min_value=0, max_value=100, value=int(grip_strength_improvement))

what_if_dash_score = st.slider("Τι θα γινοταν αν το DASH Score ήταν διαφορετικό;", 
                                min_value=0, max_value=100, value=int(dash_score_6months))

# Δημιουργία what-if input δεδομένων
what_if_input = input_data.copy()
what_if_input["physio_sessions"] = what_if_physio_sessions
what_if_input["grip_strength_improvement"] = what_if_grip_strength
what_if_input["dash_score_6months"] = what_if_dash_score

# What-if Πρόβλεψη
what_if_prediction = model.predict(what_if_input)[0]

st.info(f"📊 Εκτιμώμενος χρόνος αποκατάστασης με τις νέες τιμές: **{what_if_prediction:.2f} εβδομάδες**")


if st.button("📉 Σύγκριση με αρχική πρόβλεψη"):
    if 'initial_prediction' in st.session_state:
        delta = what_if_prediction - st.session_state['initial_prediction']
        sign = "αύξηση" if delta > 0 else "μείωση"
        st.info(f"Η νέα πρόβλεψη δείχνει {abs(delta):.2f} εβδομάδες {sign} σε σχέση με την αρχική πρόβλεψη.")
    else:
        st.warning("⚠️ Παρακαλώ κάνε πρώτα την αρχική πρόβλεψη.")

st.header("📊 Ανάλυση Υποομάδων")

selected_group = st.selectbox("Επιλογή Μεταβλητής Υποομάδας", ["sex", "treatment_type", "early_physiotherapy", "osteoporosis", "diabetes", "fracture_type"])

category_labels = {
    "sex": {0: "Άνδρας", 1: "Γυναίκα"},
    "treatment_type": {0: "Συντηρητική", 1: "Χειρουργική"},
    "early_physiotherapy": {0: "Όχι", 1: "Ναι"},
    "osteoporosis": {0: "Όχι", 1: "Ναι"},
    "diabetes": {0: "Όχι", 1: "Ναι"},
    "fracture_type": {0: "Απλό", 1: "Σύνθετο", 2: "Ενδοαρθρικό"}
}


# === Έλεγχος στήλης ===
if selected_group not in df.columns:
    st.error(f"❌ Η στήλη '{selected_group}' δεν υπάρχει στο dataset.")
    st.stop()

if df[selected_group].isnull().all():
    st.warning(f"⚠️ Όλα τα δεδομένα για '{selected_group}' είναι κενά.")
    st.stop()

# === Αντιμετώπιση τύπων ===
# Αντικατάσταση string κατηγορικών τιμών με αριθμούς
if selected_group == "sex":
    df[selected_group] = df[selected_group].map({"male": 0, "female": 1})
elif selected_group == "treatment_type":
    df[selected_group] = df[selected_group].map({"Συντηρητική": 0, "Χειρουργική": 1})
elif selected_group == "fracture_type":
    df[selected_group] = df[selected_group].map({
        "Απλό": 0, "Σύνθετο": 1, "Ενδοαρθρικό": 2
    })

# Μετατροπή σε int για ομοιομορφία
df[selected_group] = df[selected_group].astype(int)


# === Υπολογισμός μέσου χρόνου ===
group_means = df.groupby(selected_group)["recovery_time_weeks"].mean().reset_index()

# === Μετάφραση κατηγοριών ===
if selected_group in category_labels:
    group_means[selected_group] = group_means[selected_group].map(category_labels[selected_group])

# === Γράφημα ===
fig = go.Figure(data=[
    go.Bar(
        x=group_means[selected_group],
        y=group_means["recovery_time_weeks"],
        marker_color='teal'
    )
])

fig.update_layout(
    title=f"Μέσος Χρόνος Ανά Ομάδα για: {selected_group}",
    xaxis_title=selected_group,
    yaxis_title="Μέσος Χρόνος Αποκατάστασης (εβδομάδες)"
)

st.plotly_chart(fig)
