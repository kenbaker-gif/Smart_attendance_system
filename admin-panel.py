import streamlit as st
import pandas as pd
from app import dbmodule # Retain this for dbmodule.get_all_attendance()
# REMOVED: from app.dbmodule import get_all_attendance <-- Not needed

st.set_page_config(page_title="📊 Attendance Admin Panel", layout="wide")
st.title("📊 Attendance Admin Panel")

def is_verified(val):
    """Return True if value indicates successful verification."""
    if not val:
        return False
    return str(val).strip().lower() == "success"

# ----------------------------------------------
# STEP 1: INITIALIZE all_records (FIX for NameError)
# This ensures it's defined even if the DB connection fails.
# ----------------------------------------------
all_records = []

# -----------------------
# Fetch all attendance records
# -----------------------
st.subheader("All Attendance Records")
try:
    # Fetch data and assign it to the initialized variable
    all_records = dbmodule.get_all_attendance()
    
    if all_records:
        df_all = pd.DataFrame(all_records)
        st.dataframe(df_all)

        # Optional CSV export
        csv_all = df_all.to_csv(index=False).encode("utf-8")
        st.download_button("Download All Records CSV", csv_all, "attendance_all.csv", "text/csv")
    else:
        st.info("No attendance records found.")
except Exception as e:
    st.error(f"Failed to fetch attendance records: {e}")


# -----------------------
# Fetch attendance for a specific student (Search block is fine)
# -----------------------
st.subheader("Search Attendance by Student ID")
student_id_input = st.text_input("Enter Student ID:")

if st.button("Fetch Student Records") and student_id_input:
    try:
        student_records = dbmodule.get_attendance_by_student(student_id_input)
        if student_records:
            df_student = pd.DataFrame(student_records)
            st.dataframe(df_student)

            # CSV export for specific student
            csv_student = df_student.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Student Records CSV",
                csv_student,
                f"attendance_{student_id_input}.csv",
                "text/csv"
            )
        else:
            st.info("No records found for this student.")
    except Exception as e:
        st.error(f"Failed to fetch student records: {e}")

# ----------------------------------------------------------------------------------
# STATISTICS SECTION
# all_records is now guaranteed to be either the fetched list or an empty list []
# REMOVED: all_records = get_all_attendance()
# ----------------------------------------------------------------------------------

# Count verified/unverified
total_verified = sum(1 for r in all_records if is_verified(r.get("verified")))
total_unverified = sum(1 for r in all_records if not is_verified(r.get("verified")))

st.write(f"**Total Verified (Success):** {total_verified}")
st.write(f"**Total Unverified (Failure):** {total_unverified}")

# Verified per student
by_student = {}
for r in all_records:
    sid = str(r.get("student_id", "Unknown")).strip()
    if is_verified(r.get("verified")):
        by_student[sid] = by_student.get(sid, 0) + 1

if by_student:
    st.write("**Verified Students Count per Student ID:**")
    by_student_df = pd.DataFrame(
        list(by_student.items()),
        columns=["Student ID", "Verified Count"]
    )
    st.bar_chart(by_student_df.set_index("Student ID"))
else:
    st.info("No verified student attendance data to display.")