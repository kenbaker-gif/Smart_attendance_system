import streamlit as st
import pandas as pd
import os
from app import dbmodule

# 1. PAGE CONFIG & SECURITY
st.set_page_config(page_title="📊 Attendance Admin Panel", layout="wide")

def check_password():
    """Returns True if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == os.getenv("ADMIN_SECRET", "default_pass"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input("Admin Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error.
        st.text_input("Admin Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
    st.title("📊 Attendance Admin Panel")

    # 2. CACHED DATA FETCHING
    @st.cache_data(ttl=600)  # Cache results for 10 minutes
    def get_data():
        try:
            return dbmodule.get_all_attendance()
        except Exception as e:
            st.error(f"Database Error: {e}")
            return []

    all_records = get_data()

    # 3. LAYOUT: USE COLUMNS FOR STATS
    if all_records:
        df = pd.DataFrame(all_records)
        
        # Clean 'verified' column once
        df['is_success'] = df['verified'].apply(lambda x: str(x).strip().lower() == "success")

        # Top Row Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("Verified (Success)", df['is_success'].sum())
        col3.metric("Unverified", len(df) - df['is_success'].sum())

        st.divider()

        # 4. TABS FOR BETTER UX
        tab1, tab2, tab3 = st.tabs(["📋 Full Data", "🔍 Search", "📈 Analytics"])

        with tab1:
            st.subheader("All Attendance Records")
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Full CSV", csv, "attendance_all.csv", "text/csv")

        with tab2:
            st.subheader("Find Student")
            sid = st.text_input("Enter Student ID to filter:")
            if sid:
                filtered_df = df[df['student_id'].astype(str).str.contains(sid)]
                st.dataframe(filtered_df)

        with tab3:
            st.subheader("Attendance Visualization")
            # Grouping data for the chart
            success_only = df[df['is_success']]
            if not success_only.empty:
                chart_data = success_only.groupby('student_id').size().reset_index(name='Verified Count')
                st.bar_chart(chart_data.set_index('student_id'))
            else:
                st.warning("No verified records to plot.")
    else:
        st.info("No records found in the database.")

    # Refresh Button (Clears cache)
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()