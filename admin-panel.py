# admin-panel.py
import streamlit as st
import requests
import pandas as pd

API_URL = "https://lovely-imagination-production.up.railway.app"  # no trailing slash

st.title("🔐 Smart Attendance - Admin Panel")

# Admin key input
ADMIN_KEY = st.text_input("Enter Admin Key", type="password")
if not ADMIN_KEY:
    st.warning("Please enter the admin key to continue.")
    st.stop()

headers = {"Authorization": f"Bearer {ADMIN_KEY}"}

# Attendance records
st.subheader("Attendance Records")
try:
    r = requests.get(f"{API_URL}/admin/attendance", headers=headers)
    if r.status_code == 403:
        st.error("❌ Invalid Admin Key")
        st.stop()
    r.raise_for_status()
    data = r.json()
    if not data:
        st.info("No attendance records available.")
        st.stop()
    df = pd.DataFrame(data)
except Exception as e:
    st.error(f"❌ Failed to fetch data: {e}")
    st.stop()

st.dataframe(df, use_container_width=True)

# Summary
st.subheader("📊 Attendance Summary")
try:
    summary_r = requests.get(f"{API_URL}/admin/attendance_summary", headers=headers)
    summary_r.raise_for_status()
    summary = summary_r.json()
except Exception as e:
    st.error(f"❌ Failed to fetch summary: {e}")
    st.stop()

st.write(f"**Total Present:** {summary.get('total_present', 0)}")
st.write(f"**Total Absent:** {summary.get('total_absent', 0)}")

# Attendance by student
st.subheader("📈 Attendance by Student")
by_student = summary.get("by_student", {})
if by_student:
    by_student_df = pd.DataFrame(list(by_student.items()), columns=["Student", "Present Count"])
    st.bar_chart(by_student_df.set_index("Student"))
else:
    st.info("No student statistics available.")

# Download CSV
st.download_button(
    "Download Attendance CSV",
    df.to_csv(index=False).encode("utf-8"),
    "attendance_report.csv",
    "text/csv"
)
