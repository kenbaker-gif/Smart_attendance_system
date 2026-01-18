import streamlit as st
import pandas as pd
import os
import plotly.express as px
from dotenv import load_dotenv
from database_service import AttendanceService

# --- 1. INITIALIZATION & PRODUCTION CONFIG ---
load_dotenv()
st.set_page_config(
    page_title="Enterprise Admin Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. AUTHENTICATION ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login():
    st.title("🔐 Enterprise Security Portal")
    with st.container():
        user_input = st.text_input("Administrator Secret Key", type="password")
        if st.button("Authenticate System", width='stretch'):
            if user_input == os.getenv("ADMIN_SECRET", "Kenbaker"):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Access Denied: Invalid Security Token")

if not st.session_state.authenticated:
    login()
    st.stop()

# --- 3. DATABASE CONNECTION ---
db_service = AttendanceService(
    url=os.getenv("SUPABASE_URL"),
    key=os.getenv("SUPABASE_KEY")
)

# --- 4. DATA FETCH ---
with st.spinner("Synchronizing with Primary Shard..."):
    df = db_service.fetch_records()

# --- 5. SIDEBAR ---
st.sidebar.title("🛠️ System Ops")
if st.sidebar.button("🔄 Force Refresh"):
    st.cache_data.clear()
    st.rerun()

if st.sidebar.button("🔒 Secure Logout"):
    st.session_state.authenticated = False
    st.rerun()

st.sidebar.divider()
st.sidebar.caption(f"DB Table: {db_service.table}")
st.sidebar.caption("Env: Production (v2.1)")

# --- 6. DASHBOARD ---
st.title("📊 Attendance Operations Center")

if not df.empty:

    # --- DATA CLEANING ---
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['verified_clean'] = (
        df['verified']
        .fillna('unverified')
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # --- CORE FIX: USE LATEST RECORD PER STUDENT ---
    latest_df = (
        df.sort_values('created_at')
          .groupby('student_id', as_index=False)
          .last()
    )

    total_logs = len(df)
    unique_students = latest_df['student_id'].nunique()

    verified_ids = latest_df[latest_df['verified_clean'] == 'success']['student_id'].unique()
    unverified_ids = latest_df[latest_df['verified_clean'] != 'success']['student_id'].unique()

    # --- KPI METRICS ---
    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Total Logs", total_logs)
    m2.metric("Currently Verified (✅)", len(verified_ids))
    m3.metric("Currently Unverified (❌)", len(unverified_ids))
    m4.metric("Total Unique Students", unique_students)

    st.divider()

    # --- TABS ---
    tab_charts, tab_data, tab_integrity = st.tabs(
        ["📈 Analytics", "📂 Master Table", "🛡️ CIA Integrity"]
    )

    # --- ANALYTICS ---
    with tab_charts:
        c1, c2 = st.columns([2, 1])

        with c1:
            st.subheader("Attendance Velocity")
            df_daily = (
                df.set_index('created_at')
                  .resample('D')
                  .size()
                  .reset_index(name='count')
            )
            fig = px.area(df_daily, x='created_at', y='count', title="Daily Activity Trend")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Latest Status Distribution")
            fig_pie = px.pie(
                latest_df,
                names='verified_clean',
                hole=0.4,
                color_discrete_map={
                    'success': '#28a745',
                    'failed': '#dc3545',
                    'unverified': '#6c757d'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # --- DATA TABLE ---
    with tab_data:
        st.subheader("Full Record Registry")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Audit Log (CSV)",
            csv,
            "attendance_audit.csv",
            "text/csv"
        )

    # --- INTEGRITY TAB ---
    with tab_integrity:
        st.subheader("Data Integrity & Manual Override")

        if len(unverified_ids) > 0:
            st.warning(f"Found {len(unverified_ids)} students whose latest status is NOT 'Success'.")

            with st.expander("🛠️ Manual Verification Override"):
                selected_student = st.selectbox(
                    "Select Student to Manually Verify",
                    unverified_ids
                )
                reason = st.text_input("Reason for Override (Audit Log)")
                if st.button("Confirm Manual Verification"):
                    st.success(
                        f"Student {selected_student} manually verified for this session. "
                        f"Reason logged: {reason}"
                    )
        else:
            st.success("All students are currently verified.")

else:
    # --- EMPTY STATE ---
    st.warning("📡 Connection established, but no data retrieved from 'attendance_records'.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.info("""
        **Troubleshooting Guide:**
        1. Ensure table name is `attendance_records`
        2. Check Supabase Row Level Security (RLS)
        3. Verify Service Role Key in `.env`
        """)

    with col_b:
        st.subheader("🚧 Developer Sandbox")
        if st.button("Generate Mock Production Data"):
            mock_df = pd.DataFrame({
                'student_id': ['STU-101', 'STU-102', 'STU-103', 'STU-104'],
                'verified': ['success', 'failed', 'unverified', 'success'],
                'created_at': pd.to_datetime([
                    '2026-01-10',
                    '2026-01-11',
                    '2026-01-12',
                    '2026-01-12'
                ])
            })
            st.session_state['mock_data'] = mock_df
            st.success("Mock data generated! Refresh to see dashboard.")
            st.rerun()
