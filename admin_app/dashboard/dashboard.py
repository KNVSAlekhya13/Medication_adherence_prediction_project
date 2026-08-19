import sys
from pathlib import Path

# Allow imports from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.preprocessing import load_data, build_patient_features, validate_patients
from modules.prediction import predict_risk, predict_risk_batch
from modules.factor_analysis import analyze_factors, rank_factors
from modules.pattern_detection import detect_patterns
from modules.trend_forecasting import forecast_adherence, trend_direction, backtest_forecast
from modules.intervention import recommend_intervention, intervention_level
from modules.what_if import simulate_improvement
from modules.anomaly_detection import detect_anomalies
from modules.explainable_ai import explain_prediction
from modules.report_generator import generate_report
from utils.admin_auth import authenticate, AUTH_FILE


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Medication Adherence Dashboard",
    page_icon="💊",
    layout="wide"
)

px.defaults.template = "plotly_white"

# Same palette as the patient app (patient_app/static/style.css) so the
# admin console reads as one product, not a bolted-on analytics tool.
TEAL = "#0d967e"        # primary brand
TEAL_LIGHT = "#16a085"
NAVY = "#17324d"
NAVY_SOFT = "#173e60"
RED = "#c64d46"
GREEN = "#16865f"
ORANGE = "#a86b00"
PURPLE = "#6c5ce7"
BG = "#f5f8fb"

BRAND_COLORS = [TEAL, NAVY_SOFT, RED, ORANGE, PURPLE]
RISK_COLOR_MAP = {"Low Risk": GREEN, "High Risk": RED, "Insufficient Data": ORANGE}

px.defaults.color_discrete_sequence = BRAND_COLORS

# ---------------------------------------------------------
# Styling — mirrors the patient app's teal/navy "MediTrack" theme
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        html, body, [class*="css"], .stApp {
            font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
        }
        .stApp { background: #f5f8fb; }
        .block-container {
            padding-top: 5rem !important;
            padding-bottom: 3rem !important;
            max-width: 1400px;
        }

        /* Keep the admin page header fully below Streamlit's top toolbar. */
        [data-testid="stAppViewContainer"] .main .block-container {
            padding-top: 5rem !important;
        }
        .eyebrow {
            display: inline-block;
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        .main-title {
            margin-top: 8px !important;
        }

        /* Hide Streamlit hosting controls from the Admin Console. */
        .stAppDeployButton,
        [data-testid="stMainMenu"],
        [data-testid="stToolbar"] button[aria-label="Deploy"],
        [data-testid="stToolbar"] [data-testid="stMainMenu"] {
            display: none !important;
        }

        /* ---------- Header ---------- */
        .brandbar { display:flex; align-items:center; gap:10px; margin-bottom:2px; }
        .brandbar .logo {
            display:inline-grid; place-items:center; width:40px; height:40px;
            background:#dff6f2; color:#15967f; border-radius:12px;
            font-size:20px; font-weight:800;
        }
        .eyebrow { font-size:11px; letter-spacing:2px; color:#15967f; font-weight:800; }
        .main-title { font-size: 32px; font-weight: 800; color:#123e63; margin: 4px 0 2px; }
        .subtitle { font-size: 15px; margin-bottom: 10px; color: #7a8998; }

        hr, div[data-testid="stDivider"] { border-color:#e3ebf2 !important; }

        .stApp h1, .stApp h2, .stApp h3 { color:#17324d; font-weight:800; }
        .stApp p, .stApp label, .stApp span { color:inherit; }
        .stApp a { color:#0d927c; }

        /* ---------- Metric cards ---------- */
        div[data-testid="stMetric"] {
            background: #fff;
            border: 1px solid #e6edf2;
            border-radius: 18px;
            padding: 16px 18px 12px 18px;
            box-shadow: 0 5px 20px #18324d08;
        }
        div[data-testid="stMetricLabel"] {
            font-weight: 800; font-size: 11px; letter-spacing: .5px;
            color: #8493a3; text-transform: uppercase;
        }
        div[data-testid="stMetricValue"] { color: #17324d; font-weight: 800; }
        div[data-testid="stMetricDelta"] { font-weight: 700; }

        /* ---------- Badges ---------- */
        .badge-pill {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 20px;
            font-weight: 800;
            font-size: 11px;
        }
        .badge-low  { background:#e4f7ee; color:#16865f; }
        .badge-high { background:#fff0ef; color:#c64d46; }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: #fff; border-right: 1px solid #e3ebf2;
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
        .sidebar-brand { display:flex; align-items:center; gap:8px; font-size:20px; font-weight:800; color:#123e63; }
        .sidebar-brand span {
            display:inline-grid; place-items:center; background:#dff6f2; color:#15967f;
            border-radius:10px; width:32px; height:32px; font-size:16px;
        }
        .sidebar-muted { font-size:11px; color:#8b9aab; margin:4px 0 18px 40px; }
        .sidebar-heading {
            font-size:11px; letter-spacing:1.5px; color:#15967f; font-weight:800;
            text-transform:uppercase; margin:18px 0 6px;
        }

        /* Nav radio styled like the patient app's <nav> links */
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            padding: 10px 12px; border-radius: 12px; margin: 2px 0;
            transition: background .15s ease;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: #e8f7f4;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label div p {
            font-weight: 700; color: #708094; font-size: 14px;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: #e8f7f4;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div p {
            color: #0c8c79;
        }

        /* ---------- Buttons ---------- */
        .stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {
            border: 0; background: #0d967e; color: #fff; border-radius: 10px;
            font-weight: 700; padding: 10px 16px; box-shadow: none;
        }
        .stButton>button:hover, .stDownloadButton>button:hover, .stFormSubmitButton>button:hover {
            background: #0b8069; color: #fff;
        }

        /* ---------- Tabs ---------- */
        button[data-baseweb="tab"] {
            font-weight: 700; color: #708094; border-radius: 10px 10px 0 0;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #0c8c79;
        }
        div[data-baseweb="tab-highlight"] { background-color: #0d967e !important; }

        /* ---------- Inputs / sliders ---------- */
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
            border-radius: 9px !important;
        }
        div[data-testid="stSlider"] div[role="slider"] { background-color: #0d967e; }
        div[data-baseweb="slider"] > div > div { background: #0d967e !important; }

        /* ---------- Dataframes ---------- */
        div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border:1px solid #e6edf2; }

        /* ---------- Alerts ---------- */
        div[data-testid="stAlertContentInfo"] { color:#15628f; }
        div[data-testid="stAlertContentSuccess"] { color:#16865f; }
        div[data-testid="stAlertContentError"] { color:#c64d46; }
        div[data-testid="stAlertContentWarning"] { color:#a86b00; }

        /* ---------- Bordered containers (cards) ---------- */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) {
            border-radius: 18px !important;
        }

        /* ---------- Login card ---------- */
        .login-wrap { display:flex; justify-content:center; padding-top: 4vh; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #fff;
            border: 1px solid #e2ebef !important;
            border-radius: 24px !important;
            box-shadow: 0 25px 70px #17324d18;
        }
        .login-icon {
            width:46px; height:46px; border-radius:14px; display:grid; place-items:center;
            background:#dff6f2; color:#15967f; font-weight:900; font-size:21px; margin-bottom:14px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Admin credentials
# ---------------------------------------------------------

if not st.session_state.get("admin_authenticated", False):
    st.markdown(
        '<div class="brandbar" style="justify-content:center;margin-top:6vh;">'
        '<span class="logo">✚</span>'
        '<span style="font-size:23px;font-weight:800;color:#123e63;">MediTrack</span>'
        '</div>'
        '<div class="sidebar-muted" style="text-align:center;margin-left:0;">'
        'Medication Adherence Intelligence System</div>',
        unsafe_allow_html=True
    )

    left, mid, right = st.columns([1, 1.3, 1])
    with mid:
        with st.container(border=True):
            st.markdown('<div class="login-icon">🛡️</div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:26px;font-weight:800;color:#17324d;margin-bottom:2px;">'
                'Admin Login</div>'
                '<div style="color:#81909f;margin-bottom:18px;">'
                'Sign in to open the adherence intelligence dashboard.</div>',
                unsafe_allow_html=True
            )
            with st.form("admin_login"):
                email = st.text_input("Admin email", placeholder="admin@meditrack.local")
                password = st.text_input("Admin password", type="password", placeholder="••••••••")
                submit = st.form_submit_button("Admin Login", use_container_width=True)
            if submit:
                if authenticate(email, password):
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    if not AUTH_FILE.exists():
                        st.error("No admin account is configured. Run setup_admin.bat from the project root first.")
                    else:
                        st.error("Invalid admin credentials.")
    st.stop()

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
@st.cache_data
def load_project_data():
    return load_data(PROJECT_ROOT / "data")


try:
    patients, medications, adherence = load_project_data()

    # Keep every patient in the admin selector, including newly uploaded
    # patients who do not have adherence history yet.
    features = build_patient_features(adherence, patients, cutoff_date=pd.Timestamp.today().normalize())

    patient_index = patients[["patient_id"]].copy()
    features = patient_index.merge(features, on="patient_id", how="left")

    numeric_feature_columns = [
        "mean_adherence",
        "hist_adherence",
        "last_adherence",
        "total_scheduled",
        "total_taken",
        "late_doses",
        "days_recorded",
        "history_days",
        "missed_doses",
        "miss_rate",
        "recent_missed",
    ]
    for column in numeric_feature_columns:
        if column in features.columns:
            features[column] = pd.to_numeric(
                features[column], errors="coerce"
            ).fillna(0)

except Exception as error:
    st.error("Unable to load project data.")
    st.code(str(error))
    st.stop()


@st.cache_data
def compute_population_risk(features_df):
    """Vectorized risk prediction for every patient, cached until data changes."""
    risk_df = predict_risk_batch(features_df)
    return pd.concat(
        [features_df.reset_index(drop=True), risk_df.reset_index(drop=True)],
        axis=1
    )


features_with_risk = compute_population_risk(features)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    '<span class="eyebrow">ADMIN CONSOLE</span>'
    '<div class="main-title">Medication Adherence Intelligence System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Monitor adherence, estimate non-adherence risk, identify patterns, forecast trends, '
    'and generate personalized adherence-support actions.'
    '</div>',
    unsafe_allow_html=True
)

st.info("Research/demo system: model outputs are decision-support estimates, not medical diagnoses. The bundled dataset is synthetic/demo data.")

st.divider()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.markdown(
    '<div class="sidebar-brand"><span>✚</span> MediTrack</div>'
    '<div class="sidebar-muted">Admin Console</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown('<div class="sidebar-heading">Navigation</div>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Patient Analysis",
        "Risk Prediction",
        "Trend Forecasting",
        "Pattern & Anomaly Detection",
        "What-If Analysis",
        "Reports"
    ],
    label_visibility="collapsed",
)

# ---------------------------------------------------------
# Patient filters — narrow the patient selector interactively
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-heading">🔎 Filter Patients</div>', unsafe_allow_html=True)

condition_options = sorted(
    [c for c in features_with_risk.get("condition", pd.Series(dtype=str)).dropna().unique()]
)
selected_conditions = st.sidebar.multiselect(
    "Condition",
    condition_options,
    default=condition_options,
    help="Show only patients with the selected condition(s)."
)

risk_filter = st.sidebar.select_slider(
    "Risk level",
    options=["All", "Low Risk only", "High Risk only"],
    value="All",
)

search_term = st.sidebar.text_input(
    "Search patient ID",
    placeholder="e.g. P001"
).strip().upper()

filtered_features = features_with_risk.copy()

if "condition" in filtered_features.columns and selected_conditions:
    filtered_features = filtered_features[
        filtered_features["condition"].isin(selected_conditions)
        | filtered_features["condition"].isna()
    ]

if risk_filter == "Low Risk only":
    filtered_features = filtered_features[filtered_features["risk_label"] == "Low Risk"]
elif risk_filter == "High Risk only":
    filtered_features = filtered_features[filtered_features["risk_label"] == "High Risk"]

if search_term:
    filtered_features = filtered_features[
        filtered_features["patient_id"].astype(str).str.upper().str.contains(search_term)
    ]

if filtered_features.empty:
    st.sidebar.warning("No patients match these filters — showing the full list instead.")
    filtered_features = features_with_risk.copy()

patient_ids = filtered_features["patient_id"].astype(str).tolist()

if not patient_ids:
    st.error("No patients are available. Upload a patient CSV to continue.")
    st.stop()

selected_patient = st.sidebar.selectbox(
    f"Select Patient ({len(patient_ids)} match filters)",
    patient_ids
)

# ---------------------------------------------------------
# Research dataset management — deliberately isolated from operational patient DB.
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-heading">📤 Upload Research Patient</div>', unsafe_allow_html=True)
st.sidebar.caption("Uploads are added only to the research/demo CSV dataset, never to the patient application database.")
uploaded_patients = st.sidebar.file_uploader("Upload Research Patients CSV", type=["csv"], key="sidebar_research_patient_upload",
    help="Required columns: patient_id, age, gender, condition, reminder_enabled.")
if uploaded_patients is not None:
    upload_bytes=uploaded_patients.getvalue()
    upload_hash=__import__("hashlib").sha256(upload_bytes).hexdigest()
    if st.session_state.get("last_research_patient_upload_hash") != upload_hash:
        try:
            from io import StringIO
            new_patients=validate_patients(pd.read_csv(StringIO(upload_bytes.decode("utf-8-sig"))))
            research_patients_path=PROJECT_ROOT/"admin_app"/"data"/"patients.csv"
            current=pd.read_csv(research_patients_path) if research_patients_path.exists() else pd.DataFrame(columns=new_patients.columns)
            existing=set(current["patient_id"].astype(str)) if "patient_id" in current.columns else set()
            dup=sorted(set(new_patients["patient_id"].astype(str)) & existing)
            if dup: raise ValueError("Patient IDs already in research dataset: "+", ".join(dup[:20]))
            pd.concat([current,new_patients],ignore_index=True).to_csv(research_patients_path,index=False)
            st.session_state["last_research_patient_upload_hash"]=upload_hash
            st.cache_data.clear()
            st.sidebar.success(f"Added {len(new_patients)} research patient(s).")
            st.rerun()
        except Exception as upload_error:
            st.sidebar.error(f"Research upload failed: {upload_error}")

research_patients_path=PROJECT_ROOT/"admin_app"/"data"/"patients.csv"
if research_patients_path.exists():
    st.sidebar.download_button("⬇ Export Research Patients CSV", research_patients_path.read_bytes(),
                               file_name="research_patients.csv", mime="text/csv")

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-heading">📤 Upload New Patient</div>', unsafe_allow_html=True)
st.sidebar.caption(
    "Upload one or more new patient rows using the provided CSV template."
)

uploaded_patients = st.sidebar.file_uploader(
    "Upload Patients CSV",
    type=["csv"],
    key="sidebar_patient_upload",
    help="Required columns: patient_id, age, gender, condition, reminder_enabled."
)

if uploaded_patients is not None:
    upload_bytes = uploaded_patients.getvalue()
    upload_hash = __import__("hashlib").sha256(upload_bytes).hexdigest()
    if st.session_state.get("last_patient_upload_hash") != upload_hash:
        try:
            from io import StringIO
            new_patients = pd.read_csv(StringIO(upload_bytes.decode("utf-8-sig")))
            new_patients.columns = [str(column).strip() for column in new_patients.columns]
            new_patients = validate_patients(new_patients)
            initialize_database(import_csv=True)
            conn = get_connection()
            existing_ids = {r[0] for r in conn.execute("SELECT patient_id FROM patients").fetchall()}
            duplicate_existing = sorted(set(new_patients["patient_id"]) & existing_ids)
            if duplicate_existing:
                raise ValueError("Upload rejected because these patient IDs already exist: " + ", ".join(duplicate_existing[:20]))
            for _, r in new_patients.iterrows():
                conn.execute("INSERT INTO patients(patient_id,age,gender,condition,reminder_enabled) VALUES(?,?,?,?,?)",
                             (str(r.patient_id), int(r.age), str(r.gender), str(r.condition), int(r.reminder_enabled)))
                conn.execute("INSERT INTO audit_log(actor_type,actor_id,action,entity_type,entity_id,details) VALUES(?,?,?,?,?,?)",
                             ("admin", "admin", "import_patient", "patient", str(r.patient_id), "CSV upload"))
            conn.commit(); conn.close()
            st.session_state["last_patient_upload_hash"] = upload_hash
            st.cache_data.clear()
            st.sidebar.success(f"✓ {len(new_patients)} new patient(s) added to the shared database.")
            st.rerun()
        except Exception as upload_error:
            st.sidebar.error(f"Upload failed: {upload_error}")

# Database-backed export: CSV is an import/export format, not the source of truth.
try:
    initialize_database(import_csv=True)
    export_bytes = export_patients_csv(Path("patients_export.csv"))
    export_path = Path("patients_export.csv")
    with open(export_path, "rb") as fh:
        st.sidebar.download_button("⬇ Export Patients CSV", fh.read(), file_name="patients_export.csv", mime="text/csv")
    export_path.unlink(missing_ok=True)
except Exception:
    pass

st.sidebar.markdown("---")
if st.sidebar.button("↪ Logout", use_container_width=True):
    st.session_state.admin_authenticated = False
    st.rerun()


selected_matches = features_with_risk[
    features_with_risk["patient_id"].astype(str) == str(selected_patient)
]

if selected_matches.empty:
    st.error("Selected patient could not be loaded.")
    st.stop()

selected_row = selected_matches.iloc[0]

selected_patient_info = patients[
    patients["patient_id"] == selected_patient
]

selected_medications = medications[
    medications["patient_id"] == selected_patient
]

patient_history = adherence[
    adherence["patient_id"] == selected_patient
].copy()

patient_history["date"] = pd.to_datetime(
    patient_history["date"]
)

patient_history["adherence_rate"] = (
    patient_history["taken_doses"] /
    patient_history["scheduled_doses"]
).fillna(0).clip(0, 1)

if patient_history.empty:
    st.sidebar.info(
        "This patient has no adherence history yet. "
        "Prediction uses the currently available patient features."
    )


def risk_badge(label):
    css_class = "badge-low" if label == "Low Risk" else "badge-high" if label == "High Risk" else "badge-pill"
    icon = "🟢" if label == "Low Risk" else "🔴" if label == "High Risk" else "🟠"
    st.markdown(
        f'<span class="badge-pill {css_class}">{icon} {label}</span>',
        unsafe_allow_html=True
    )


def risk_gauge(probability, title="Risk Probability"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%"},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": NAVY_SOFT},
            "steps": [
                {"range": [0, 40], "color": "#e4f7ee"},
                {"range": [40, 70], "color": "#fff2dc"},
                {"range": [70, 100], "color": "#fff0ef"},
            ],
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":

    st.header("📊 Overview Dashboard")

    tab_patient, tab_population = st.tabs(["👤 Patient View", "🌍 Population Insights"])

    with tab_patient:
        total_patients = len(features_with_risk)
        average_adherence = features_with_risk["mean_adherence"].mean()
        high_risk_count = int((features_with_risk["risk_label"] == "High Risk").sum())
        total_missed = int(
            adherence["scheduled_doses"].sum()
            - adherence["taken_doses"].sum()
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Patients", total_patients)
        col2.metric("Average Adherence", f"{average_adherence:.1%}")
        col3.metric(
            "High-Risk Patients",
            high_risk_count,
            delta=f"{high_risk_count / total_patients:.0%} of patients",
            delta_color="inverse"
        )
        col4.metric("Total Missed Doses", total_missed)

        st.divider()

        header_col, badge_col = st.columns([4, 1])
        with header_col:
            st.subheader(f"Patient Overview — {selected_patient}")
        with badge_col:
            risk_badge(selected_row["risk_label"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Patient Adherence", f"{selected_row['mean_adherence']:.1%}")
        col2.metric("Missed Doses", int(selected_row["missed_doses"]))
        col3.metric("Late Doses", int(selected_row["late_doses"]))

        st.subheader("Adherence Trend")

        if patient_history.empty:
            st.info("No adherence history recorded for this patient yet.")
        else:
            fig = px.area(
                patient_history,
                x="date",
                y="adherence_rate",
                markers=True,
                title=f"Daily Adherence — {selected_patient}",
                color_discrete_sequence=[BRAND_COLORS[0]],
            )
            fig.add_hline(
                y=0.80, line_dash="dash", line_color=RED,
                annotation_text="80% target", annotation_position="top left"
            )
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            fig.update_layout(
                xaxis_title="Date", yaxis_title="Adherence",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_population:
        st.caption(
            f"Showing {len(filtered_features)} of {len(features_with_risk)} patients "
            "based on the sidebar filters."
        )

        pop_col1, pop_col2 = st.columns([1, 1])

        with pop_col1:
            risk_counts = filtered_features["risk_label"].value_counts().reset_index()
            risk_counts.columns = ["risk_label", "count"]
            fig_pie = px.pie(
                risk_counts,
                names="risk_label",
                values="count",
                title="Risk Level Distribution",
                color="risk_label",
                color_discrete_map=RISK_COLOR_MAP,
                hole=0.45,
            )
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        with pop_col2:
            fig_hist = px.histogram(
                filtered_features,
                x="mean_adherence",
                nbins=25,
                title="Adherence Distribution",
                color_discrete_sequence=[BRAND_COLORS[1]],
            )
            fig_hist.update_xaxes(tickformat=".0%", title="Mean Adherence")
            fig_hist.update_yaxes(title="Patient Count")
            st.plotly_chart(fig_hist, use_container_width=True)

        if "condition" in filtered_features.columns:
            fig_box = px.box(
                filtered_features.dropna(subset=["condition"]),
                x="condition",
                y="mean_adherence",
                color="condition",
                title="Adherence by Condition",
                points="outliers",
                color_discrete_sequence=BRAND_COLORS,
            )
            fig_box.update_yaxes(tickformat=".0%", title="Mean Adherence")
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        if "age" in filtered_features.columns:
            fig_scatter = px.scatter(
                filtered_features,
                x="age",
                y="mean_adherence",
                color="risk_label",
                color_discrete_map=RISK_COLOR_MAP,
                size="miss_rate",
                size_max=18,
                hover_data=["patient_id", "condition"] if "condition" in filtered_features.columns else ["patient_id"],
                title="Age vs. Adherence (bubble size = miss rate)",
            )
            fig_scatter.update_yaxes(tickformat=".0%", title="Mean Adherence")
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("🚨 Top 10 Highest-Risk Patients")
        display_cols = ["patient_id", "age", "mean_adherence", "risk_label", "risk_probability"]
        if "condition" in filtered_features.columns:
            display_cols.insert(1, "condition")
        top_risk = (
            filtered_features.sort_values("risk_probability", ascending=False)
            .head(10)[display_cols]
        )
        st.dataframe(
            top_risk.style.format({
                "mean_adherence": "{:.1%}",
                "risk_probability": "{:.1%}",
            }),
            use_container_width=True
        )


# =========================================================
# PATIENT ANALYSIS
# =========================================================
elif page == "Patient Analysis":

    st.header("👤 Patient Analysis")

    if not selected_patient_info.empty:

        patient = selected_patient_info.iloc[0]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Patient ID", patient["patient_id"])
        col2.metric("Age", int(patient["age"]))
        col3.metric("Gender", patient["gender"])
        col4.metric("Condition", patient["condition"])
        with col5:
            risk_badge(selected_row["risk_label"])

    st.subheader("Medications")

    if selected_medications.empty:
        st.info("No medication records found.")
    else:
        st.dataframe(selected_medications, use_container_width=True)

    factor_col, ranking_col = st.columns([1, 1])

    with factor_col:
        st.subheader("Adherence Factors")
        factors = analyze_factors(selected_row)
        for factor in factors:
            st.write("•", factor)

    with ranking_col:
        st.subheader("Factor Ranking")
        ranked = rank_factors(selected_row)
        ranking_df = pd.DataFrame(ranked, columns=["Factor", "Impact"])
        ranking_df["Impact"] = ranking_df["Impact"].round(3)

        fig_factors = px.bar(
            ranking_df.sort_values("Impact"),
            x="Impact",
            y="Factor",
            orientation="h",
            title="Relative Factor Impact",
            color="Impact",
            color_continuous_scale=["#e4f7ee", RED],
        )
        fig_factors.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_factors, use_container_width=True)


# =========================================================
# RISK PREDICTION
# =========================================================
elif page == "Risk Prediction":

    # Model performance (5-fold cross-validation on the included demo dataset)
    try:
        metrics_file = PROJECT_ROOT / "models" / "model_metrics.csv"
        metrics = pd.read_csv(metrics_file).iloc[0]
        st.subheader("📈 Model Performance")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Accuracy", f"{metrics['accuracy_percent']:.2f}%")
        mc2.metric("Precision", f"{metrics['precision_percent']:.2f}%")
        mc3.metric("Recall", f"{metrics['recall_percent']:.2f}%")
        mc4.metric("F1 Score", f"{metrics['f1_percent']:.2f}%")
        mc5.metric("ROC-AUC", f"{metrics['roc_auc_percent']:.2f}%")
        st.caption(
            "Evaluated with 5-fold stratified cross-validation on the included "
            "demonstration dataset. Not clinical validation data."
        )
    except Exception:
        pass

    st.divider()
    st.header("🤖 Medication Non-Adherence Risk Prediction")

    try:
        prediction = predict_risk(selected_row.to_dict())

        gauge_col, info_col = st.columns([1, 1])

        with gauge_col:
            if prediction.get("prediction_available"):
                st.plotly_chart(risk_gauge(prediction["risk_probability"]), use_container_width=True)
            else:
                st.warning(prediction["message"])

        with info_col:
            risk_badge(prediction["risk_label"])
            if prediction.get("prediction_available"):
                st.metric("Predicted Non-Adherence Risk", f"{prediction['risk_probability']:.1%}")
                st.metric("Predicted Future Adherence ≥ 70%", f"{prediction['good_adherence_probability']:.1%}")
            else:
                st.caption(prediction["message"])

        explanation = explain_prediction(selected_row, prediction)

        st.subheader("🧠 Local Model Explanation")
        st.info(explanation["explanation"])

        st.subheader("Recommended Intervention")
        level = intervention_level(selected_row)
        st.write(f"**Priority:** {level}")

        recommendations = recommend_intervention(selected_row)
        for recommendation in recommendations:
            st.write("•", recommendation)

    except Exception as error:
        st.error("Model prediction could not be completed.")
        st.code(str(error))


# =========================================================
# TREND FORECASTING
# =========================================================
elif page == "Trend Forecasting":

    st.header("📈 Adherence Trend Forecasting")

    direction = trend_direction(adherence, selected_patient)
    backtest = backtest_forecast(adherence, selected_patient)
    if backtest.get("mae") is not None:
        bt1, bt2, bt3 = st.columns(3)
        bt1.metric("Backtest MAE", f"{backtest['mae']:.3f}")
        bt2.metric("Backtest RMSE", f"{backtest['rmse']:.3f}")
        bt3.metric("Backtest days", str(backtest['n_test']))
        st.caption("Linear-regression baseline evaluated by rolling historical backtesting; not a clinical forecast.")
    trend_icon = {"Improving": "📈", "Declining": "📉", "Stable": "➖"}.get(direction, "❔")
    st.metric("Predicted Trend", f"{trend_icon} {direction}")

    forecast_periods = st.slider(
        "Forecast horizon (days)", min_value=3, max_value=21, value=7, step=1
    )

    forecast = forecast_adherence(adherence, selected_patient, periods=forecast_periods)

    if forecast.empty:
        st.warning("Not enough data to generate a forecast.")
    else:
        history = patient_history[["date", "adherence_rate"]].rename(
            columns={"adherence_rate": "adherence"}
        )
        history["series"] = "Historical"

        future = forecast[["date", "forecast_adherence"]].rename(
            columns={"forecast_adherence": "adherence"}
        )
        future["series"] = "Forecast"

        combined = pd.concat([history, future], ignore_index=True)

        st.subheader(f"Next {forecast_periods} Days")
        st.dataframe(forecast, use_container_width=True)

        fig = px.line(
            combined,
            x="date",
            y="adherence",
            color="series",
            markers=True,
            title=f"Adherence History & {forecast_periods}-Day Forecast",
            color_discrete_map={"Historical": BRAND_COLORS[1], "Forecast": BRAND_COLORS[0]},
        )
        fig.update_traces(
            selector=dict(name="Forecast"), line=dict(dash="dash")
        )
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        fig.update_layout(hovermode="x unified", xaxis_title="Date", yaxis_title="Adherence")
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PATTERN & ANOMALY DETECTION
# =========================================================
elif page == "Pattern & Anomaly Detection":

    st.header("🔍 Pattern & Anomaly Detection")

    st.subheader("Detected Patterns")

    patterns = detect_patterns(adherence)
    patient_patterns = patterns[patterns["patient_id"] == selected_patient]

    if patient_patterns.empty:
        st.success("No significant patterns detected.")
    else:
        st.dataframe(patient_patterns, use_container_width=True)

    st.subheader("🚨 Anomalies")

    anomalies = detect_anomalies(patient_history)
    detected_anomalies = anomalies[anomalies["anomaly"] == True]  # noqa: E712

    if not anomalies.empty and "date" in anomalies.columns:
        fig_anomaly = px.scatter(
            anomalies,
            x="date",
            y="adherence_rate",
            color="anomaly",
            color_discrete_map={True: RED, False: TEAL},
            title=f"Adherence Timeline with Anomalies — {selected_patient}",
            labels={"anomaly": "Anomaly"},
        )
        fig_anomaly.update_yaxes(range=[0, 1], tickformat=".0%")
        fig_anomaly.update_traces(marker=dict(size=10))
        st.plotly_chart(fig_anomaly, use_container_width=True)

    if detected_anomalies.empty:
        st.success("No unusual adherence records detected.")
    else:
        st.warning(f"{len(detected_anomalies)} unusual record(s) detected.")
        st.dataframe(detected_anomalies, use_container_width=True)


# =========================================================
# WHAT-IF ANALYSIS
# =========================================================
elif page == "What-If Analysis":

    st.header("🔮 What-If Adherence Scenario")

    current_adherence = float(selected_row["mean_adherence"])
    st.metric("Current Adherence", f"{current_adherence:.1%}")

    improvement = st.slider(
        "Expected improvement", min_value=0.0, max_value=0.50, value=0.10, step=0.05
    )

    simulation = simulate_improvement(selected_row, improvement, predictor=predict_risk)

    col1, col2, col3 = st.columns(3)
    col1.metric("Current", f"{simulation['current_adherence']:.1%}")
    col2.metric("Projected", f"{simulation['projected_adherence']:.1%}")
    col3.metric("Improvement", f"{simulation['improvement']:.1%}")
    if simulation.get("model_recomputed"):
        r1, r2 = st.columns(2)
        r1.metric("Current model risk", f"{simulation['current_risk']:.1%}")
        r2.metric("Scenario model risk", f"{simulation['projected_risk']:.1%}")

    fig_whatif = go.Figure(data=[
        go.Bar(
            x=["Current", "Projected"],
            y=[simulation["current_adherence"], simulation["projected_adherence"]],
            marker_color=[BRAND_COLORS[1], BRAND_COLORS[0]],
            text=[f"{simulation['current_adherence']:.0%}", f"{simulation['projected_adherence']:.0%}"],
            textposition="outside",
        )
    ])
    fig_whatif.update_yaxes(range=[0, 1], tickformat=".0%", title="Adherence")
    fig_whatif.update_layout(title="Current vs. Projected Adherence", height=350)
    st.plotly_chart(fig_whatif, use_container_width=True)

    st.info(
        "This simulation shows a hypothetical adherence improvement. "
        "It is not a clinical prediction."
    )


# =========================================================
# REPORTS
# =========================================================
elif page == "Reports":

    st.header("📄 Patient Report Generator")

    st.write(f"Generate a PDF report for **{selected_patient}**.")

    if st.button("Generate PDF Report", type="primary"):

        try:
            prediction = predict_risk(selected_row.to_dict())
            recommendations = recommend_intervention(selected_row)

            report_path = generate_report(
                selected_patient,
                selected_row,
                prediction,
                recommendations,
                output_dir=PROJECT_ROOT / "reports"
            )

            report_file = Path(report_path)

            st.success("Patient report generated successfully.")

            with open(report_file, "rb") as file:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=file,
                    file_name=report_file.name,
                    mime="application/pdf"
                )

        except Exception as error:
            st.error("Unable to generate the report.")
            st.code(str(error))


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "Medication Adherence Intelligence System | "
    "Educational / decision-support prototype"
)
