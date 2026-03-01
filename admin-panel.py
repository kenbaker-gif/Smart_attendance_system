import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Attendance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Base */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #e8e8f0;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: #0a0a0f;
}

[data-testid="stHeader"] { background: transparent; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e30;
}

/* Title */
.dash-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00f5c4;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1;
}

.dash-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #4a4a6a;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 24px 0;
}

.metric-card {
    background: #0f0f1a;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}

.metric-card.green::before  { background: #00f5c4; }
.metric-card.blue::before   { background: #4a9eff; }
.metric-card.orange::before { background: #ff8c42; }
.metric-card.purple::before { background: #a78bfa; }

.metric-card:hover { border-color: #2e2e50; }

.metric-label {
    font-size: 0.72rem;
    color: #4a4a6a;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 8px;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #e8e8f0;
    line-height: 1;
}

.metric-sub {
    font-size: 0.75rem;
    color: #4a4a6a;
    margin-top: 6px;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #4a4a6a;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e1e30;
    padding-bottom: 8px;
    margin: 28px 0 16px 0;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-success { background: rgba(0,245,196,0.1); color: #00f5c4; border: 1px solid rgba(0,245,196,0.2); }
.badge-failed  { background: rgba(255,80,80,0.1);  color: #ff5050;  border: 1px solid rgba(255,80,80,0.2); }

/* Institution filter pills */
.pill {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 500;
    cursor: pointer;
    border: 1px solid #1e1e30;
    background: #0f0f1a;
    color: #4a4a6a;
    margin-right: 8px;
}

/* Dataframe overrides */
[data-testid="stDataFrame"] {
    border: 1px solid #1e1e30 !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* Input */
[data-testid="stTextInput"] input {
    background: #0f0f1a !important;
    border: 1px solid #1e1e30 !important;
    color: #e8e8f0 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div {
    background: #0f0f1a !important;
    border: 1px solid #1e1e30 !important;
    border-radius: 8px !important;
}

/* Download button */
[data-testid="stDownloadButton"] button {
    background: #00f5c4 !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
}

/* Divider */
hr { border-color: #1e1e30 !important; }

/* Spinner */
.stSpinner > div { border-top-color: #00f5c4 !important; }

</style>
""", unsafe_allow_html=True)

# ── Config ─────────────────────────────────────────────────────────────────
API_URL = "https://smartattendancemvp-production.up.railway.app"

# ── Header ─────────────────────────────────────────────────────────────────
col_title, col_auth = st.columns([3, 1])

with col_title:
    st.markdown('<p class="dash-title">SMART ATTENDANCE</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="dash-subtitle">Admin Dashboard · {datetime.now().strftime("%d %b %Y")}</p>', unsafe_allow_html=True)

with col_auth:
    ADMIN_KEY = st.text_input("Admin Key", type="password", placeholder="Bearer token", label_visibility="collapsed")

if not ADMIN_KEY:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Enter your admin key above to load the dashboard.")
    st.stop()

headers = {"Authorization": f"Bearer {ADMIN_KEY}"}

# ── Fetch data ─────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    try:
        r = requests.get(f"{API_URL}/admin/attendance-records", headers=headers, timeout=15)
        r.raise_for_status()
        raw = r.json()
        df = pd.DataFrame(raw) if raw else pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to fetch attendance records: {e}")
        st.stop()

    try:
        sr = requests.get(f"{API_URL}/admin/attendance_summary", headers=headers, timeout=15)
        sr.raise_for_status()
        summary = sr.json()
    except:
        summary = {}

    try:
        students_r = requests.get(f"{API_URL}/students", headers=headers, timeout=15)
        students_r.raise_for_status()
        students_data = students_r.json()
        total_students = students_data.get("count", 0) if isinstance(students_data, dict) else len(students_data)
    except:
        total_students = 0

if df.empty:
    st.warning("No attendance records found.")
    st.stop()

# ── Process data ───────────────────────────────────────────────────────────
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert('Africa/Nairobi')  # UTC+3 EAT
    df['date']      = df['timestamp'].dt.date
    df['hour']      = df['timestamp'].dt.hour

total_records  = len(df)
total_verified = len(df[df['verified'] == 'success']) if 'verified' in df.columns else 0
total_failed   = len(df[df['verified'] == 'failed'])  if 'verified' in df.columns else 0
rate           = round((total_verified / total_records * 100), 1) if total_records > 0 else 0

# ── Metric cards ───────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card green">
        <div class="metric-label">Verified Today</div>
        <div class="metric-value">{total_verified}</div>
        <div class="metric-sub">successful scans</div>
    </div>
    <div class="metric-card blue">
        <div class="metric-label">Total Students</div>
        <div class="metric-value">{total_students}</div>
        <div class="metric-sub">enrolled</div>
    </div>
    <div class="metric-card orange">
        <div class="metric-label">Failed Scans</div>
        <div class="metric-value">{total_failed}</div>
        <div class="metric-sub">unrecognised</div>
    </div>
    <div class="metric-card purple">
        <div class="metric-label">Success Rate</div>
        <div class="metric-value">{rate}%</div>
        <div class="metric-sub">recognition accuracy</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Filters ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Filter Records</div>', unsafe_allow_html=True)

col_inst, col_status, col_date = st.columns(3)

with col_inst:
    institutions = ["All"]
    if 'institution_id' in df.columns:
        institutions += sorted(df['institution_id'].dropna().unique().tolist())
    selected_inst = st.selectbox("Institution", institutions)

with col_status:
    selected_status = st.selectbox("Status", ["All", "success", "failed"])

with col_date:
    if 'date' in df.columns:
        dates = ["All"] + sorted(df['date'].dropna().unique().tolist(), reverse=True)
        selected_date = st.selectbox("Date", dates)
    else:
        selected_date = "All"

# Apply filters
filtered = df.copy()
if selected_inst != "All" and 'institution_id' in filtered.columns:
    filtered = filtered[filtered['institution_id'] == selected_inst]
if selected_status != "All" and 'verified' in filtered.columns:
    filtered = filtered[filtered['verified'] == selected_status]
if selected_date != "All" and 'date' in filtered.columns:
    filtered = filtered[filtered['date'] == selected_date]

# ── Charts ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Analytics</div>', unsafe_allow_html=True)

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("**Scans by Hour**")
    if 'hour' in filtered.columns and not filtered.empty:
        hourly = filtered.groupby('hour').size().reset_index(name='count')
        hourly = hourly.set_index('hour').reindex(range(24), fill_value=0)
        st.bar_chart(hourly, color="#00f5c4", height=200)
    else:
        st.info("No data")

with col_chart2:
    st.markdown("**Attendance by Student**")
    if 'student_id' in filtered.columns and not filtered.empty:
        by_student = filtered[filtered['verified'] == 'success'].groupby('student_id').size().reset_index(name='count')
        by_student = by_student.sort_values('count', ascending=False).head(10).set_index('student_id')
        st.bar_chart(by_student, color="#4a9eff", height=200)
    else:
        st.info("No data")

# ── Records table ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Attendance Records</div>', unsafe_allow_html=True)

display_cols = ['student_id', 'verified', 'confidence', 'institution_id', 'timestamp']
display_cols = [c for c in display_cols if c in filtered.columns]

st.dataframe(
    filtered[display_cols].sort_values('timestamp', ascending=False) if 'timestamp' in filtered.columns else filtered[display_cols],
    use_container_width=True,
    height=400,
    column_config={
        "verified":       st.column_config.TextColumn("Status"),
        "confidence":     st.column_config.NumberColumn("Confidence", format="%.2f"),
        "institution_id": st.column_config.TextColumn("Institution"),
        "timestamp":      st.column_config.DatetimeColumn("Time", format="DD MMM YYYY, HH:mm"),
    }
)

st.markdown(f"<p style='color:#4a4a6a;font-size:0.75rem;'>{len(filtered)} records shown</p>", unsafe_allow_html=True)

# ── Export ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)

col_dl, col_info = st.columns([1, 3])
with col_dl:
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "↓ Download CSV",
        csv,
        f"attendance_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv"
    )
with col_info:
    st.markdown(f"<p style='color:#4a4a6a;font-size:0.8rem;margin-top:8px;'>Exporting {len(filtered)} filtered records</p>", unsafe_allow_html=True)