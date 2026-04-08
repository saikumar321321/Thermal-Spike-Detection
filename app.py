import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thermal – Spike Detection",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:     #080d18;
    --card:   #0f1826;
    --card2:  #17243a;
    --cyan:   #00e5ff;
    --amber:  #ffab00;
    --red:    #ff3d57;
    --green:  #00e676;
    --muted:  #607d8b;
    --text:   #dde4f0;
    --border: rgba(0,229,255,0.13);
}

html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0b1422 0%,#080d18 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

h1,h2,h3 { font-family:'Rajdhani',sans-serif !important; letter-spacing:1px; }

[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 14px !important;
}

.stSelectbox>div>div, .stNumberInput>div>div>input,
.stSlider>div, .stTextInput>div>div>input {
    background: var(--card2) !important;
    border-radius: 6px !important;
}

.stButton>button {
    font-family:'Rajdhani',sans-serif !important;
    font-size:16px !important; font-weight:700 !important;
    letter-spacing:2px !important;
    background: linear-gradient(135deg,#005f64,#007a82) !important;
    color:#e0f7fa !important;
    border:1px solid var(--cyan) !important;
    border-radius:6px !important;
    padding:10px 36px !important;
    text-transform:uppercase;
    white-space:nowrap !important;
}
.stButton>button:hover {
    background:linear-gradient(135deg,#007a82,#00a0ad) !important;
    box-shadow:0 0 22px rgba(0,229,255,.3) !important;
}

hr { border-color: var(--border) !important; }

.spike-box {
    background:linear-gradient(135deg,rgba(255,61,87,.14),rgba(255,61,87,.04));
    border:1px solid var(--red); border-left:4px solid var(--red);
    border-radius:10px; padding:22px 26px; font-family:'Rajdhani',sans-serif;
}
.safe-box {
    background:linear-gradient(135deg,rgba(0,230,118,.14),rgba(0,230,118,.04));
    border:1px solid var(--green); border-left:4px solid var(--green);
    border-radius:10px; padding:22px 26px; font-family:'Rajdhani',sans-serif;
}

.mono { font-family:'Share Tech Mono',monospace; }

.sec-hdr {
    font-family:'Rajdhani',sans-serif; font-size:12px; font-weight:700;
    letter-spacing:3px; text-transform:uppercase; color:var(--cyan);
    margin-bottom:10px; padding-bottom:5px; border-bottom:1px solid var(--border);
}
.page-title {
    font-family:'Rajdhani',sans-serif; font-size:38px; font-weight:700;
    color:var(--cyan); letter-spacing:2px; margin-bottom:2px;
}
.page-sub { color:var(--muted); font-size:14px; margin-bottom:24px; }
.sec-title {
    font-family:'Rajdhani',sans-serif; font-size:22px; font-weight:700;
    color:var(--amber); letter-spacing:1px;
    margin:28px 0 10px; padding-bottom:6px;
    border-bottom:2px solid rgba(255,171,0,.25);
}

.icard {
    background:var(--card); border:1px solid var(--border);
    border-radius:10px; padding:22px 26px; margin-bottom:14px;
    line-height:1.78; font-size:14.5px; color:#b0bec5;
}
.icard p  { margin:0 0 8px; }
.icard ul { padding-left:20px; margin:0; }
.icard ul li { margin-bottom:5px; }

.kpi-box {
    background:var(--card2); border:1px solid var(--border);
    border-radius:8px; padding:16px; text-align:center;
}
.kpi-val { font-family:'Share Tech Mono',monospace; font-size:26px; color:var(--cyan); }
.kpi-lbl { font-size:11px; color:var(--muted); letter-spacing:1.5px; margin-top:4px; text-transform:uppercase; }

.step {
    display:flex; align-items:flex-start; gap:16px;
    padding:14px 0; border-bottom:1px solid var(--border);
}
.step-num {
    width:34px; height:34px; border-radius:50%; flex-shrink:0;
    background:linear-gradient(135deg,#005f64,#007a82);
    display:flex; align-items:center; justify-content:center;
    font-family:'Share Tech Mono',monospace; font-size:13px; color:#e0f7fa;
}
.step-title { font-family:'Rajdhani',sans-serif; font-size:17px; font-weight:700; color:var(--text); }
.step-desc  { font-size:13px; color:var(--muted); margin-top:2px; line-height:1.65; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        with open("ohe.pkl",     "rb") as f: ohe_cols      = pickle.load(f)
        with open("scaler.pkl",  "rb") as f: all_feat_cols = pickle.load(f)
        with open("Xgboost.pkl", "rb") as f: model         = pickle.load(f)
        return ohe_cols, all_feat_cols, model, None
    except Exception as e:
        return None, None, None, str(e)

ohe_cols, all_feat_cols, model, load_err = load_artifacts()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATASET FOR CHARTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("Dataset.csv")
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df, None
    except Exception as e:
        return None, str(e)

dataset, dataset_err = load_dataset()

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
NUMERIC_COLS = [
    "user_session_duration_s","user_request_count","user_payload_size_mb",
    "user_cpu_cores_used","user_gpu_memory_used_gb","user_ram_used_gb",
    "user_disk_io_mbps","user_power_draw_w","user_cpu_contribution_pct",
    "user_gpu_contribution_pct","user_heat_contribution_pct",
    "inlet_temp_c","outlet_temp_c","hotspot_temp_c","cooling_capacity_pct",
    "airflow_rate_cfm","ambient_temp_c","humidity_pct","rolling_avg_temp_15m_c",
]
TIME_COLS  = ["hour","day_of_week","month","is_weekend"]
WORK_TYPES = ["analytics","batch","data-pipeline","ETL","inference","ML-training","stream","web"]
DCZ_ZONES  = ["Zone-A","Zone-B","Zone-C","Zone-D"]
SERVER_IDS = [f"SRV-{i:04d}" for i in range(1, 201)]
USER_IDS   = [f"USR-{i:05d}" for i in range(1, 4921)]
RACK_IDS   = [f"RACK-{i:03d}" for i in range(1, 51)]

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,24,38,0.6)",
    font_color="#dde4f0",
    font_family="Inter, sans-serif",
    margin=dict(t=50, b=40, l=50, r=20),
)

# Reusable axis/legend style dicts (applied individually, not via **PLOT_LAYOUT)
_XAXIS = dict(
    gridcolor="rgba(0,229,255,0.07)",
    linecolor="rgba(0,229,255,0.2)",
    tickfont=dict(size=11, color="#607d8b"),
)
_YAXIS = dict(
    gridcolor="rgba(0,229,255,0.07)",
    linecolor="rgba(0,229,255,0.2)",
    tickfont=dict(size=11, color="#607d8b"),
)
_LEGEND = dict(
    bgcolor="rgba(15,24,38,0.8)",
    bordercolor="rgba(0,229,255,0.2)",
    borderwidth=1,
    font=dict(size=12),
)

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_input(inp):
    row = {col: inp.get(col, 0.0) for col in NUMERIC_COLS}
    for col in all_feat_cols:
        if col not in row:
            row[col] = 0
    for key in [
        f"ServerID_{inp['ServerID'].lower()}",
        f"UserID_{inp['UserID'].lower()}",
        f"DataCentreZone_{inp['DataCentreZone'].lower()}",
        f"WorkType_{inp['WorkType'].lower()}",
    ]:
        if key in row:
            row[key] = 1
    ts = inp.get("Timestamp", datetime.datetime.now())
    if isinstance(ts, str):
        ts = pd.to_datetime(ts)
    row["hour"]        = ts.hour
    row["day_of_week"] = ts.weekday()
    row["month"]       = ts.month
    row["is_weekend"]  = int(ts.weekday() >= 5)
    ordered = list(all_feat_cols) + TIME_COLS
    return np.array([[row.get(c, 0) for c in ordered]])


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:14px 0 28px;'>
        <div style='font-family:Rajdhani,sans-serif;font-size:27px;font-weight:700;
                    color:#00e5ff;letter-spacing:3px;'>🌡️ ThermalSentinel  </div>
        <div style='font-family:Share Tech Mono,monospace;font-size:11px;
                    color:#607d8b;letter-spacing:2px;margin-top:5px;'>AI MONITORING SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["🏠 Home","📌 Project Overview","🔮 Predict"],
                    label_visibility="collapsed")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":

    st.markdown("<div class='page-title'>Thermal Spike Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Real-time AI-powered monitoring for data centre thermal events</div>", unsafe_allow_html=True)
    st.markdown("---")

    hero_l, hero_r = st.columns([1.15, 1], gap="large")

    with hero_l:
        st.markdown("""
        <div class="icard">
            <p>
                A <b style='color:#00e5ff;'>thermal spike</b> is a sudden, dangerous rise in
                server temperature that shoots beyond the safe operating threshold — typically
                <b style='color:#ff3d57;'>above 85 °C at the hotspot</b>. Unlike gradual
                heat build-up, spikes are characterised by <em>rapid onset</em> and
                <em>extreme peak intensity</em>.
            </p>
            <p>
                Modern hyperscale data centres generate millions of telemetry readings every
                minute. Manual monitoring is no longer feasible —
                <b style='color:#00e5ff;'>Thermal Spike Detector</b> uses a trained
                <b style='color:#ffab00;'>XGBoost classifier</b> to predict thermal spikes
                <em>before they escalate</em>, giving operators time to intervene.
            </p>
            <p>
                The system reads <b style='color:#00e5ff;'>24 real-time sensor signals</b>
                spanning workload metrics, rack temperatures, cooling efficiency, and
                environmental conditions to deliver an instant binary risk assessment.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with hero_r:
        st.markdown("""
        <div style='background:#0f1826;border:1px solid rgba(0,229,255,0.15);
                    border-radius:12px;padding:24px;text-align:center;'>
            <svg viewBox="0 0 320 230" xmlns="http://www.w3.org/2000/svg" width="100%">
                <rect x="55" y="15" width="210" height="185" rx="7"
                      fill="#080d18" stroke="#00e5ff" stroke-width="1.5"/>
                <rect x="68" y="28"  width="184" height="20" rx="3" fill="#17243a" stroke="#00e5ff"  stroke-width="0.7"/>
                <rect x="68" y="54"  width="184" height="20" rx="3" fill="#17243a" stroke="#00e5ff"  stroke-width="0.7"/>
                <rect x="68" y="80"  width="184" height="20" rx="3" fill="#17243a" stroke="#00e5ff"  stroke-width="0.7"/>
                <rect x="68" y="106" width="184" height="20" rx="3" fill="#17243a" stroke="#ffab00"  stroke-width="1.4"/>
                <rect x="68" y="132" width="184" height="20" rx="3" fill="#17243a" stroke="#00e5ff"  stroke-width="0.7"/>
                <rect x="68" y="158" width="184" height="20" rx="3" fill="#17243a" stroke="#00e5ff"  stroke-width="0.7"/>
                <circle cx="238" cy="38"  r="4" fill="#00e676"/>
                <circle cx="224" cy="38"  r="4" fill="#00e676"/>
                <circle cx="238" cy="64"  r="4" fill="#00e676"/>
                <circle cx="224" cy="64"  r="4" fill="#00e676"/>
                <circle cx="238" cy="90"  r="4" fill="#00e676"/>
                <circle cx="224" cy="90"  r="4" fill="#00e676"/>
                <circle cx="238" cy="142" r="4" fill="#00e676"/>
                <circle cx="224" cy="142" r="4" fill="#00e676"/>
                <circle cx="238" cy="168" r="4" fill="#00e676"/>
                <circle cx="224" cy="168" r="4" fill="#00e676"/>
                <circle cx="238" cy="116" r="4" fill="#ff3d57">
                    <animate attributeName="opacity" values="1;0.1;1" dur="0.85s" repeatCount="indefinite"/>
                </circle>
                <circle cx="224" cy="116" r="4" fill="#ffab00">
                    <animate attributeName="opacity" values="1;0.2;1" dur="0.65s" repeatCount="indefinite"/>
                </circle>
                <path d="M267 116 Q278 103 290 116 Q302 129 314 116" stroke="#ff3d57" stroke-width="1.8" fill="none">
                    <animate attributeName="opacity" values="0.8;0.1;0.8" dur="1.1s" repeatCount="indefinite"/>
                </path>
                <path d="M267 116 Q280 99 294 116 Q308 133 322 116" stroke="#ffab00" stroke-width="1.2" fill="none">
                    <animate attributeName="opacity" values="0.45;0.05;0.45" dur="1.4s" repeatCount="indefinite"/>
                </path>
                <text x="160" y="213" text-anchor="middle"
                      font-family="Share Tech Mono,monospace" font-size="11"
                      fill="#607d8b" letter-spacing="2">SERVER RACK · ZONE-B</text>
            </svg>
            <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                        color:#ff3d57;margin-top:6px;letter-spacing:2px;">
                ⚠ THERMAL ANOMALY DETECTED
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kpis = [("20,000","TOTAL RECORDS"),("15.0%","SPIKE RATE"),("200","SERVERS"),
            ("4","DC ZONES"),("8","WORK TYPES"),("24","SENSOR SIGNALS")]
    for col,(val,lbl) in zip([k1,k2,k3,k4,k5,k6], kpis):
        col.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-val">{val}</div>
            <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-title'>⚡ What is a Thermal Spike?</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="icard">
        <p>
            Thermal spikes occur when a server's internal temperature rises sharply beyond its
            rated operating range. They differ from gradual heat build-up because of their
            <em>sudden onset</em> and <em>high peak intensity</em>. Even a brief spike can:
        </p>
        <ul>
            <li>Trigger CPU/GPU thermal throttling — degrading application performance instantly</li>
            <li>Cause emergency shutdowns — leading to unplanned, costly downtime</li>
            <li>Accelerate hardware aging — dramatically shortening component lifespan</li>
            <li>Lead to permanent data loss — especially during write-intensive workloads like ETL or ML-training</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sec-title'>🔥 Factors Affecting Thermal Spikes</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    causes = [
        ("#00e5ff","🖥️ Compute Overload",
         ["Sudden GPU/CPU saturation","ML-training or inference burst jobs",
          "High payload & disk I/O spikes","Session duration overruns"]),
        ("#ffab00","❄️ Cooling Failure",
         ["Reduced airflow through rack","CRAC unit or chiller trip",
          "Clogged air filters","Cooling capacity drops below demand"]),
        ("#00e676","🌍 Environmental",
         ["High ambient temperature","Humidity excursions",
          "Poor rack-level airflow design","Adjacent server heat bleeding"]),
    ]
    for col,(clr,title,items) in zip([c1,c2,c3], causes):
        li = "".join(f"<li>{i}</li>" for i in items)
        col.markdown(f"""
        <div class="icard" style='border-top:3px solid {clr};'>
            <div style='font-family:Rajdhani,sans-serif;font-size:17px;font-weight:700;
                        color:{clr};margin-bottom:8px;'>{title}</div>
            <ul style='font-size:13px;'>{li}</ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-title'>🤖 How Thermal Spike Detector Predicts</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="icard">
        <p>At prediction time, the operator enters current server telemetry into the
        <b style='color:#00e5ff;'>🔮 Predict</b> page. Thermal Spike Detector then:</p>
        <ul>
            <li><b style='color:#dde4f0;'>Encodes</b> categorical fields (ServerID, UserID, DataCentreZone, WorkType) via one-hot encoding — producing 5,132 binary columns</li>
            <li><b style='color:#dde4f0;'>Extracts</b> temporal features (hour, day-of-week, month, is_weekend) from the timestamp</li>
            <li><b style='color:#dde4f0;'>Assembles</b> all 5,155 features in the exact order the XGBoost model was trained on</li>
            <li><b style='color:#dde4f0;'>Runs inference</b> through the XGBoost classifier in under 5 ms</li>
            <li><b style='color:#dde4f0;'>Returns</b> a binary prediction + spike risk probability shown on a live gauge chart</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(0,229,255,.07),rgba(0,122,130,.07));
                border:1px solid rgba(0,229,255,.25);border-radius:10px;
                padding:20px 28px;display:flex;align-items:center;gap:16px;'>
        <div style='font-size:38px;'>⚡</div>
        <div>
            <div style='font-family:Rajdhani,sans-serif;font-size:20px;font-weight:700;
                        color:#00e5ff;letter-spacing:1px;'>Ready to run a prediction?</div>
            <div style='color:#607d8b;font-size:14px;margin-top:4px;'>
                Head to <b style='color:#dde4f0;'>🔮 Predict</b> in the sidebar — enter your
                server telemetry and get an instant spike risk assessment powered by XGBoost.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📌 Project Overview":

    st.markdown("<div class='page-title'>Project Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Dataset · Feature engineering · Model training & selection</div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── KPI strip ─────────────────────────────────────────────────────────────
    k1,k2,k3,k4,k5 = st.columns(5)
    kpis = [("5,155","TOTAL FEATURES"),("XGBoost","FINAL MODEL"),
            ("20,000","RECORDS"),("80 / 20","TRAIN / TEST"),("3","MODELS TRAINED")]
    for col,(val,lbl) in zip([k1,k2,k3,k4,k5], kpis):
        col.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-val">{val}</div>
            <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  SPIKE TREND CHART  (from Dataset.csv)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='sec-title'>📊 Thermal Spike Trends — Dataset Analysis</div>", unsafe_allow_html=True)

    if dataset is not None:
        df = dataset.copy()
        df['date']        = df['Timestamp'].dt.date
        df['hour_label']  = df['Timestamp'].dt.strftime('%Y-%m-%d %H:00')
        df['month_label'] = df['Timestamp'].dt.strftime('%b %Y')
        df['month_order'] = df['Timestamp'].dt.to_period('M')
        df['month_name']  = df['Timestamp'].dt.strftime('%B %Y')  # Full month name for dropdown
        df['year']        = df['Timestamp'].dt.year

        # ── Toggle buttons ──────────────────────────────────────────────────
        if 'spike_view' not in st.session_state:
            st.session_state['spike_view'] = 'Daily'
        if 'selected_month' not in st.session_state:
            st.session_state['selected_month'] = None
        if 'selected_date' not in st.session_state:
            st.session_state['selected_date'] = None

        st.markdown("""
        <style>
        div[data-testid="column"] button {
            width: 100% !important;
            font-family: 'Rajdhani', sans-serif !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            letter-spacing: 3px !important;
            border-radius: 6px !important;
            padding: 10px 0 !important;
            white-space: nowrap !important;
        }
        </style>
        """, unsafe_allow_html=True)

        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("🕐 Day", use_container_width=True):
                st.session_state['spike_view'] = 'Hourly'
                st.session_state['selected_date'] = None
        with btn_col2:
            if st.button("📅 Month", use_container_width=True):
                st.session_state['spike_view'] = 'Daily'
                st.session_state['selected_month'] = None
        with btn_col3:
            if st.button("📊 Year", use_container_width=True):
                st.session_state['spike_view'] = 'Yearly'

        view = st.session_state['spike_view']

        # ── Additional controls based on view ────────────────────────────────
        if view == 'Hourly':
            selected_date = st.date_input(
                "Select Date for Hourly Analysis:",
                value=df['date'].max(),
                min_value=df['date'].min(),
                max_value=df['date'].max()
            )
            st.session_state['selected_date'] = selected_date
            
        elif view == 'Daily':
            unique_months = sorted(df['month_name'].unique(), 
                                 key=lambda x: pd.to_datetime(x, format='%B %Y'))
            selected_month = st.selectbox(
                "Select Month for Daily Analysis:",
                options= unique_months,
                index=0
            )
            st.session_state['selected_month'] = selected_month if selected_month != 'All Months' else None

        # ── Aggregate based on selected view ────────────────────────────────
        if view == 'Hourly':
            filtered_df = df[df['date'] == st.session_state['selected_date']].copy()
            if filtered_df.empty:
                st.warning("No data available for selected date.")
                agg = pd.DataFrame()
            else:
                agg = (
                    filtered_df.groupby('hour_label')
                              .agg(total=('thermal_spike_label','count'),
                                   spikes=('thermal_spike_label','sum'))
                              .reset_index()
                              .sort_values('hour_label')
                )
                agg['normal']     = agg['total'] - agg['spikes']
                agg['spike_rate'] = (agg['spikes'] / agg['total'] * 100).round(2)
                x_col   = 'hour_label'
                x_label = f'Hour - {st.session_state["selected_date"]}'
                title   = f'Hourly Thermal Spikes - {st.session_state["selected_date"]}'
                tick_angle = -45
                nticks     = 24

        elif view == 'Daily':
            if st.session_state['selected_month'] is None:
                # Show all daily data
                agg = (
                    df.groupby('date')
                      .agg(total=('thermal_spike_label','count'),
                           spikes=('thermal_spike_label','sum'))
                      .reset_index()
                      .sort_values('date')
                )
                agg['date_str'] = agg['date'].astype(str)
                x_col = 'date_str'
                month_filter_text = "All Months"
            else:
                # Filter by selected month
                month_df = df[df['month_name'] == st.session_state['selected_month']].copy()
                agg = (
                    month_df.groupby('date')
                            .agg(total=('thermal_spike_label','count'),
                                 spikes=('thermal_spike_label','sum'))
                            .reset_index()
                            .sort_values('date')
                )
                agg['date_str'] = agg['date'].astype(str)
                x_col = 'date_str'
                month_filter_text = st.session_state['selected_month']
            
            agg['normal']     = agg['total'] - agg['spikes']
            agg['spike_rate'] = (agg['spikes'] / agg['total'] * 100).round(2)
            x_label = 'Date'
            title   = f'Daily Thermal Spikes - {month_filter_text}'
            tick_angle = -40
            nticks     = 31

        else:  # Yearly (Full-year)
            agg = (
                df.groupby(['year', 'month_order', 'month_label'])
                  .agg(total=('thermal_spike_label','count'),
                       spikes=('thermal_spike_label','sum'))
                  .reset_index()
                  .sort_values(['year', 'month_order'])
            )
            agg['normal']     = agg['total'] - agg['spikes']
            agg['spike_rate'] = (agg['spikes'] / agg['total'] * 100).round(2)
            agg['x_label'] = agg['year'].astype(str) + ' ' + agg['month_label']
            
            x_col   = 'x_label'
            x_label = 'Month (Year)'
            title   = 'Full-Year Thermal Spike Trends'
            tick_angle = -30
            nticks     = 24

        # ── Active view indicator ────────────────────────────────────────────
        view_info = f"{view} "
        if view == 'Hourly':
            view_info += f"({st.session_state.get('selected_date', 'No date')})"
        elif view == 'Daily' and st.session_state.get('selected_month'):
            view_info += f"({st.session_state['selected_month'][:8]}...)"
        
        st.markdown(f"""
        <div style='display:inline-block;background:linear-gradient(135deg,rgba(0,229,255,0.12),rgba(0,229,255,0.04));
                    border:1px solid rgba(0,229,255,0.35);border-radius:20px;
                    padding:4px 18px;margin:8px 0 14px;font-family:Share Tech Mono,monospace;
                    font-size:12px;color:#00e5ff;letter-spacing:2px;'>
            ● VIEWING: {view_info.upper()}
        </div>
        """, unsafe_allow_html=True)

        # ── Build chart ──────────────────────────────────────────────────────
        if not agg.empty:
            fig = go.Figure()

            fig.add_trace(go.Bar(
                name="Normal",
                x=agg[x_col],
                y=agg['normal'],
                marker_color="#00e676",
                marker_line_width=0,
                opacity=0.80,
            ))
            fig.add_trace(go.Bar(
                name="Thermal Spike",
                x=agg[x_col],
                y=agg['spikes'],
                marker_color="#ff3d57",
                marker_line_width=0,
                opacity=0.90,
            ))
            fig.add_trace(go.Scatter(
                name="Spike Rate %",
                x=agg[x_col],
                y=agg['spike_rate'],
                mode="lines+markers",
                line=dict(color="#ffab00", width=2.5, dash="dot"),
                marker=dict(size=7, color="#ffab00", symbol="diamond"),
                yaxis="y2",
            ))

            y2_max = max(30, float(agg['spike_rate'].max()) * 1.3)

            fig.update_layout(
                **PLOT_LAYOUT,
                height=450 if view == 'Yearly' else 400,
                barmode="stack",
                title=dict(
                    text=title,
                    font=dict(family="Rajdhani, sans-serif", size=17, color="#00e5ff"),
                    x=0.01,
                ),
                xaxis=dict(
                    title=dict(text=x_label, font=dict(color="#607d8b", size=12)),
                    tickangle=tick_angle,
                    nticks=nticks,
                    **_XAXIS,
                ),
                yaxis=dict(
                    title=dict(text="Record Count", font=dict(color="#607d8b", size=12)),
                    **_YAXIS,
                ),
                yaxis2=dict(
                    title=dict(text="Spike Rate (%)", font=dict(color="#ffab00", size=12)),
                    tickfont=dict(color="#ffab00", size=11),
                    overlaying="y",
                    side="right",
                    range=[0, y2_max],
                    showgrid=False,
                ),
                legend=dict(
                    **_LEGEND,
                    orientation="h",
                    yanchor="bottom", y=1.02,
                    xanchor="right",  x=1,
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

        # ── Summary KPI cards ────────────────────────────────────────────────
        total_spikes = int(df['thermal_spike_label'].sum())
        total_rec    = len(df)
        daily_agg = (
            df.groupby('date')
              .agg(spikes=('thermal_spike_label','sum'))
              .reset_index()
        )
        avg_daily = daily_agg['spikes'].mean()
        peak_day  = daily_agg.loc[daily_agg['spikes'].idxmax(), 'date']

        ic1, ic2, ic3, ic4 = st.columns(4)
        for col, val, lbl in [
            (ic1, f"{total_spikes:,}",                    "TOTAL SPIKES"),
            (ic2, f"{total_spikes/total_rec*100:.1f}%",   "OVERALL SPIKE RATE"),
            (ic3, f"{avg_daily:.0f}/day",                  "AVG DAILY SPIKES"),
            (ic4, str(peak_day),                           "PEAK SPIKE DAY"),
        ]:
            col.markdown(f"""
            <div class="kpi-box" style='margin-top:10px;'>
                <div class="kpi-val">{val}</div>
                <div class="kpi-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    else:
        st.warning(f"⚠️ Dataset.csv not found — place it alongside app.py to enable charts. ({dataset_err})")

    # ── Models Trained ────────────────────────────────────────────────────────
    st.markdown("<div class='sec-title'>🤖 Models Trained to Learn X → y</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="icard" style='margin-bottom:18px;'>
        <p>
            Three <b style='color:#00e5ff;'>classification algorithms</b> were trained and
            evaluated on the same 80/20 stratified test split, all learning the relation
            between <b style='color:#ffab00;'>5,155 input features (X)</b> and
            <b style='color:#ff3d57;'>thermal_spike_label (y)</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Performance comparison ────────────────────────────────────────────────
    st.markdown("<div class='sec-title'>📊 Model Performance Comparison</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="icard" style='margin-bottom:12px;'>
        <p style='margin:0;font-size:13px;color:#607d8b;'>
            Indicative scores on the 20% stratified hold-out test set.
            XGBoost achieved the best F1-score on the minority spike class.
        </p>
    </div>""", unsafe_allow_html=True)

    perf = pd.DataFrame({
        "Model":             ["Logistic Regression","Random Forest Classifier","⚡ XGBoost Classification ✅"],
        "Accuracy":          ["~84%","~84%","~86%"],
        "Precision (Spike)": ["~74%","~88%","~93%"],
        "Recall (Spike)":    ["~68%","~84%","~92%"],
        "F1-Score (Spike)":  ["~71%","~86%","~92%"],
        "Status":            ["Baseline","Strong","✅ FINALIZED"],
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":

    st.markdown("<div class='page-title'>Thermal Spike Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Enter server telemetry to get an instant XGBoost spike risk assessment.</div>", unsafe_allow_html=True)

    if load_err:
        st.error(
            f"⚠️ Cannot load model artefacts: {load_err}\n\n"
            "Place ohe.pkl, scaler.pkl, and Xgboost.pkl in the same folder as app.py."
        )
        st.stop()

    st.markdown('<div class="sec-hdr">🖥️ Server Identity</div>', unsafe_allow_html=True)
    ci1, ci2, ci3, ci4 = st.columns(4)
    with ci1: server_id = st.selectbox("Server ID",        SERVER_IDS)
    with ci2: rack_id   = st.selectbox("Rack ID",          RACK_IDS)
    with ci3: dcz       = st.selectbox("Data Centre Zone", DCZ_ZONES)
    with ci4: user_id   = st.selectbox("User ID",          USER_IDS)

    ct1, ct2, _ = st.columns([1, 1.2, 1.8])
    with ct1: work_type = st.selectbox("Work Type", WORK_TYPES)
    with ct2:
        ts_date = st.date_input("Date",  value=datetime.date(2024, 1, 1))
        ts_time = st.time_input("Time",  value=datetime.time(0, 0))
    selected_ts = datetime.datetime.combine(ts_date, ts_time)

    st.markdown("---")

    st.markdown('<div class="sec-hdr">⚙️ User Workload Metrics</div>', unsafe_allow_html=True)
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        session_dur   = st.number_input("Session Duration (s)",  10.0,  13891.0, 3050.0, 10.0)
        request_count = st.number_input("Request Count",         1,      6000,   1046)
    with w2:
        payload_mb    = st.number_input("Payload Size (MB)",     0.5,   7000.0,  206.0,  0.5)
        cpu_cores     = st.number_input("CPU Cores Used",        0.5,    120.0,   33.0,  0.5)
    with w3:
        gpu_mem       = st.number_input("GPU Memory (GB)",       0.0,     80.0,   18.0,  0.5)
        ram_used      = st.number_input("RAM Used (GB)",         0.5,    380.0,   85.0,  0.5)
    with w4:
        disk_io       = st.number_input("Disk I/O (MBps)",       1.0,   7000.0,  270.0,  1.0)
        power_draw    = st.number_input("Power Draw (W)",       50.0,   1540.0,  480.0,  1.0)

    s1, s2, s3 = st.columns(3)
    with s1: cpu_pct  = st.slider("CPU Contribution (%)",  0.0, 100.0, 25.8, 0.1)
    with s2: gpu_pct  = st.slider("GPU Contribution (%)",  0.0, 100.0, 22.5, 0.1)
    with s3: heat_pct = st.slider("Heat Contribution (%)", 1.0,  31.0,  9.6, 0.1)

    st.markdown("---")

    st.markdown('<div class="sec-hdr">🌡️ Thermal & Environmental Sensors</div>', unsafe_allow_html=True)
    th1, th2, th3, th4, th5 = st.columns(5)
    with th1: inlet_temp   = st.number_input("Inlet Temp (°C)",   19.0,  31.0, 25.5, 0.1)
    with th2: outlet_temp  = st.number_input("Outlet Temp (°C)",  26.0,  61.0, 41.6, 0.1)
    with th3: hotspot_temp = st.number_input("Hotspot Temp (°C)", 30.0,  95.0, 62.2, 0.1)
    with th4: cooling_cap  = st.slider("Cooling Capacity (%)",    24.0, 100.0, 71.8, 0.1)
    with th5: airflow      = st.number_input("Airflow (CFM)",   1280.0,3600.0,2606.0, 1.0)

    en1, en2, en3 = st.columns(3)
    with en1: ambient_temp = st.number_input("Ambient Temp (°C)", 16.0, 35.0, 24.7, 0.1)
    with en2: humidity     = st.slider("Humidity (%)",            31.0, 70.0, 50.1, 0.1)
    with en3: rolling_avg  = st.number_input("Rolling Avg 15m (°C)", 43.0, 83.0, 62.2, 0.1)

    st.markdown("---")

    run = st.button("⚡  PREDICT", use_container_width=True)

    if run:
        inp = {
            "ServerID": server_id,  "DataCentreZone": dcz,
            "UserID":   user_id,    "WorkType":       work_type,
            "Timestamp": selected_ts,
            "user_session_duration_s":    session_dur,
            "user_request_count":         request_count,
            "user_payload_size_mb":       payload_mb,
            "user_cpu_cores_used":        cpu_cores,
            "user_gpu_memory_used_gb":    gpu_mem,
            "user_ram_used_gb":           ram_used,
            "user_disk_io_mbps":          disk_io,
            "user_power_draw_w":          power_draw,
            "user_cpu_contribution_pct":  cpu_pct,
            "user_gpu_contribution_pct":  gpu_pct,
            "user_heat_contribution_pct": heat_pct,
            "inlet_temp_c":               inlet_temp,
            "outlet_temp_c":              outlet_temp,
            "hotspot_temp_c":             hotspot_temp,
            "cooling_capacity_pct":       cooling_cap,
            "airflow_rate_cfm":           airflow,
            "ambient_temp_c":             ambient_temp,
            "humidity_pct":               humidity,
            "rolling_avg_temp_15m_c":     rolling_avg,
        }
        try:
            X = preprocess_input(inp)
            pred   = model.predict(X)[0]
            proba  = model.predict_proba(X)[0]
            s_pct  = proba[1] * 100
            n_pct  = proba[0] * 100

            st.markdown("---")
            st.markdown('<div class="sec-hdr">📡 Prediction Result</div>', unsafe_allow_html=True)
            res_col, gauge_col = st.columns([2, 1])

            with res_col:
                if pred:
                    spike_causes = []
                    spike_solutions = []

                    if hotspot_temp >= 80:
                        spike_causes.append(f"🌡️ <b>Critical hotspot temperature</b> ({hotspot_temp}°C) — well above safe threshold of ~80°C")
                        spike_solutions.append("⬇️ Immediately lower workload on this server and activate emergency cooling mode")
                    elif hotspot_temp >= 70:
                        spike_causes.append(f"🌡️ <b>Elevated hotspot temperature</b> ({hotspot_temp}°C) — approaching critical threshold")
                        spike_solutions.append("📉 Reduce workload intensity and increase cooling capacity proactively")

                    if cooling_cap < 50:
                        spike_causes.append(f"❄️ <b>Insufficient cooling capacity</b> ({cooling_cap}%) — cooling system underperforming")
                        spike_solutions.append("🔧 Inspect CRAC units and chiller systems; check for filter blockages or coolant levels")
                    elif cooling_cap < 65:
                        spike_causes.append(f"❄️ <b>Marginal cooling capacity</b> ({cooling_cap}%) — insufficient for current heat load")
                        spike_solutions.append("⚙️ Boost cooling fan speed and verify CRAC unit output for this rack")

                    if power_draw >= 1200:
                        spike_causes.append(f"⚡ <b>Extreme power draw</b> ({power_draw} W) — generating excessive heat")
                        spike_solutions.append("🔄 Migrate high-power jobs to a cooler zone or distribute load across multiple servers")
                    elif power_draw >= 900:
                        spike_causes.append(f"⚡ <b>High power draw</b> ({power_draw} W) — elevated thermal output")
                        spike_solutions.append("📊 Throttle batch or ML-training jobs during peak thermal periods")

                    if airflow < 1800:
                        spike_causes.append(f"💨 <b>Low airflow rate</b> ({airflow} CFM) — inadequate heat dissipation from rack")
                        spike_solutions.append("🌀 Clear rack obstructions; check blanking panels and verify CRAC airflow direction")

                    if rolling_avg >= 75:
                        spike_causes.append(f"📈 <b>Sustained high rolling average</b> ({rolling_avg}°C over 15 min) — prolonged heat build-up")
                        spike_solutions.append("⏸️ Schedule a cooldown period; defer non-critical workloads for at least 15 minutes")

                    if ambient_temp >= 28:
                        spike_causes.append(f"🌍 <b>High ambient temperature</b> ({ambient_temp}°C) — room-level heat reducing cooling delta")
                        spike_solutions.append("🏢 Check data centre HVAC system; raise chilled water temperature setpoint if needed")

                    if work_type in ["ML-training", "batch", "ETL", "data-pipeline"] and power_draw >= 700:
                        spike_causes.append(f"🖥️ <b>Thermally intensive workload type</b> ({work_type}) with high resource utilisation")
                        spike_solutions.append(f"📅 Schedule {work_type} jobs during off-peak hours or in thermally cooler zones (Zone-A/B)")

                    if gpu_mem >= 60:
                        spike_causes.append(f"🎮 <b>High GPU memory utilisation</b> ({gpu_mem} GB) — GPU operating at or near thermal limits")
                        spike_solutions.append("🔁 Use GPU memory optimisation techniques (mixed precision, gradient checkpointing)")

                    if outlet_temp - inlet_temp >= 20:
                        delta = round(outlet_temp - inlet_temp, 1)
                        spike_causes.append(f"🔁 <b>Large inlet–outlet temperature delta</b> ({delta}°C) — rack absorbing excessive heat")
                        spike_solutions.append("📐 Review rack airflow layout; ensure hot/cold aisle containment is intact")

                    if not spike_causes:
                        spike_causes.append("🔍 <b>Combined sensor readings</b> indicate an elevated thermal risk based on the model's learned patterns")
                        spike_solutions.append("🛡️ Perform a full thermal audit of the server and surrounding rack environment")

                    causes_html = "".join(
                        f"<li style='margin-bottom:7px;color:#ef9a9a;'>{c}</li>"
                        for c in spike_causes
                    )
                    solutions_html = "".join(
                        f"<li style='margin-bottom:7px;color:#a5d6a7;'>{s}</li>"
                        for s in spike_solutions
                    )

                    st.markdown(f"""
                    <div class="spike-box">
                        <div style='font-size:30px;font-weight:700;color:#ff3d57;letter-spacing:2px;'>
                            ⚠️ THERMAL SPIKE DETECTED
                        </div>
                        <div style='margin-top:8px;color:#ef9a9a;font-size:15px;'>
                            Thermal spike predicted for <b>{server_id}</b> in <b>{dcz}</b>.
                            Immediate cooling intervention recommended.
                        </div>
                        <div style='margin-top:14px;font-family:Share Tech Mono,monospace;
                                    font-size:22px;color:#ff3d57;'>
                            Spike Risk: {s_pct:.1f}%
                        </div>
                        <div style='margin-top:5px;font-size:13px;color:#607d8b;'>
                            Normal probability: {n_pct:.1f}%
                        </div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    cau_col, sol_col = st.columns(2)
                    with cau_col:
                        st.markdown(f"""
                        <div style='background:linear-gradient(135deg,rgba(255,61,87,.10),rgba(255,61,87,.03));
                                    border:1px solid rgba(255,61,87,0.5);border-left:4px solid #ff3d57;
                                    border-radius:10px;padding:20px 22px;'>
                            <div style='font-family:Rajdhani,sans-serif;font-size:17px;font-weight:700;
                                        color:#ff3d57;letter-spacing:1px;margin-bottom:12px;'>
                                🔥 IDENTIFIED CAUSES
                            </div>
                            <ul style='padding-left:18px;margin:0;font-size:13.5px;line-height:1.8;'>
                                {causes_html}
                            </ul>
                        </div>""", unsafe_allow_html=True)

                    with sol_col:
                        st.markdown(f"""
                        <div style='background:linear-gradient(135deg,rgba(0,230,118,.10),rgba(0,230,118,.03));
                                    border:1px solid rgba(0,230,118,0.5);border-left:4px solid #00e676;
                                    border-radius:10px;padding:20px 22px;'>
                            <div style='font-family:Rajdhani,sans-serif;font-size:17px;font-weight:700;
                                        color:#00e676;letter-spacing:1px;margin-bottom:12px;'>
                                🛠️ SUGGESTED SOLUTIONS
                            </div>
                            <ul style='padding-left:18px;margin:0;font-size:13.5px;line-height:1.8;'>
                                {solutions_html}
                            </ul>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-box">
                        <div style='font-size:30px;font-weight:700;color:#00e676;letter-spacing:2px;'>
                            ✅ NORMAL OPERATION
                        </div>
                        <div style='margin-top:8px;color:#a5d6a7;font-size:15px;'>
                            No thermal spike predicted for <b>{server_id}</b> in <b>{dcz}</b>.
                            System operating within safe parameters.
                        </div>
                        <div style='margin-top:14px;font-family:Share Tech Mono,monospace;
                                    font-size:22px;color:#00e676;'>
                            Stability: {n_pct:.1f}%
                        </div>
                        <div style='margin-top:5px;font-size:13px;color:#607d8b;'>
                            Spike probability: {s_pct:.1f}%
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="sec-hdr">📋 Input Summary</div>', unsafe_allow_html=True)
                sm1,sm2,sm3,sm4 = st.columns(4)
                sm1.metric("Hotspot Temp",    f"{hotspot_temp}°C")
                sm2.metric("Power Draw",      f"{power_draw} W")
                sm3.metric("Cooling Cap",     f"{cooling_cap}%")
                sm4.metric("Rolling Avg",     f"{rolling_avg}°C")
                sm5,sm6,sm7,sm8 = st.columns(4)
                sm5.metric("Outlet Temp",     f"{outlet_temp}°C")
                sm6.metric("Airflow",         f"{airflow} CFM")
                sm7.metric("Heat Contrib",    f"{heat_pct}%")
                sm8.metric("Work Type",        work_type)

            with gauge_col:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=s_pct,
                    number={"suffix":"%","font":{"size":30,"color":"#dde4f0"}},
                    gauge={
                        "axis":{"range":[0,100],"tickcolor":"#607d8b",
                                "tickfont":{"color":"#607d8b","size":10}},
                        "bar":{"color":"#ff3d57" if pred else "#00e676"},
                        "bgcolor":"#17243a",
                        "borderwidth":1,"bordercolor":"rgba(0,229,255,0.2)",
                        "steps":[
                            {"range":[0,40],   "color":"rgba(0,230,118,0.12)"},
                            {"range":[40,70],  "color":"rgba(255,171,0,0.12)"},
                            {"range":[70,100], "color":"rgba(255,61,87,0.12)"},
                        ],
                        "threshold":{"line":{"color":"#ffab00","width":2},"value":50},
                    },
                    title={"text":"Spike Risk","font":{"color":"#607d8b","size":14}},
                ))
                fig.update_layout(
                    height=240, margin=dict(t=50,b=10,l=20,r=20),
                    paper_bgcolor="rgba(0,0,0,0)", font_color="#dde4f0",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div style='background:#0f1826;border:1px solid rgba(0,229,255,0.13);
                            border-radius:8px;padding:14px 16px;font-size:13px;'>
                    <div style='font-family:Rajdhani,sans-serif;font-size:12px;font-weight:700;
                                color:#607d8b;letter-spacing:2px;margin-bottom:8px;'>RISK BANDS</div>
                    <div style='margin-bottom:5px;'>
                        <span style='color:#00e676;'>■</span>
                        <span style='color:#a5d6a7;'> 0–40%</span>
                        <span style='color:#607d8b;font-size:12px;'> Low Risk</span>
                    </div>
                    <div style='margin-bottom:5px;'>
                        <span style='color:#ffab00;'>■</span>
                        <span style='color:#ffe082;'> 40–70%</span>
                        <span style='color:#607d8b;font-size:12px;'> Elevated Risk</span>
                    </div>
                    <div>
                        <span style='color:#ff3d57;'>■</span>
                        <span style='color:#ef9a9a;'> 70–100%</span>
                        <span style='color:#607d8b;font-size:12px;'> High Risk</span>
                    </div>
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")