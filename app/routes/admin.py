from fastapi import APIRouter, Depends, HTTPException
from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()  # Must be called before os.getenv

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ADMIN_SECRET = os.getenv("ADMIN_SECRET")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

router = APIRouter(
    prefix="/admin",
    tags=["Admin & Reports"]
)

# Admin verification
def verify_admin(token: str = ADMIN_SECRET):
    if token != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    return True

# Attendance endpoint
@router.get("/attendance")
def get_attendance(admin: bool = Depends(verify_admin)):
    response = supabase.table("attendance_records").select("*").execute()
    return response.data

# Attendance summary
@router.get("/attendance_summary")
def get_summary(admin: bool = Depends(verify_admin)):
    response = supabase.table("attendance_records").select("*").execute()
    rows = response.data or []

    total_present = sum(1 for r in rows if r.get("verified") == "success")
    total_absent = sum(1 for r in rows if r.get("verified") == "failed")

    by_student = {}
    for r in rows:
        student_id = r.get("student_id", "Unknown")
        by_student[student_id] = by_student.get(student_id, 0)
        if r.get("verified") == "success":
            by_student[student_id] += 1

    return {
        "total_present": total_present,
        "total_absent": total_absent,
        "by_student": by_student
    }
