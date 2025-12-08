import streamlit as st
import requests
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()  # Must be called before os.getenv

# -----------------------
# CONFIG
# -----------------------
USE_LOCAL = False # Set False to test Railway deployment
API_URL = "http://localhost:8000" if USE_LOCAL else "https://vivacious-charisma-production.up.railway.app/"

# ADMIN SECRET (make sure this is set in your environment)
ADMIN_SECRET = os.getenv("ADMIN_SECRET")
if not ADMIN_SECRET:
    st.error("ADMIN_SECRET not found in environment variables!")
    st.stop()

headers = {"Authorization": f"Bearer {ADMIN_SECRET}"}

# -----------------------
# FETCH DATA
# -----------------------
st.title("📊 Attendance Admin Panel")

# Healthcheck
try:
    health_r = requests.get(f"{API_URL}/health")
    health_r.raise_for_status()
    st.success("API Health: OK ✅")
except Exception as e:
    st.error(f"API Healthcheck Failed: {e}")
    st.stop()

# Attendance Records
st.subheader("Attendance Records")
try:
    r = requests.get(f"{API_URL}/admin/attendance", headers=headers)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if df.empty:
        st.info("No attendance records found.")
    else:
        st.dataframe(df)
except Exception as e:
    st.error(f"Failed to fetch attendance: {e}")
    st.stop()

# Summary Stats
st.subheader("Attendance Summary")
try:
    summary_r = requests.get(f"{API_URL}/admin/attendance_summary", headers=headers)
    summary_r.raise_for_status()
    summary = summary_r.json()

    st.write(f"**Total Present:** {summary.get('total_present', 0)}")
    st.write(f"**Total Absent:** {summary.get('total_absent', 0)}")

    by_student = summary.get("by_student", {})
    if by_student:
        st.write("**Attendance by Student:**")
        by_student_df = pd.DataFrame(list(by_student.items()), columns=["Student ID", "Present Count"])
        st.bar_chart(by_student_df.set_index("Student ID"))
    else:
        st.info("No student data to display.")

except Exception as e:
    st.error(f"Failed to fetch summary: {e}")
    st.stop()

# Optional: Export CSV
if not df.empty:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "attendance_report.csv", "text/csv")
