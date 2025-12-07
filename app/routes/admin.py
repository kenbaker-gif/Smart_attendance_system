# app/routes/admin.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import create_client
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

# Admin setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ADMIN_SECRET = os.getenv("ADMIN_SECRET")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials not found.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

security = HTTPBearer()

def verify_admin(token: HTTPAuthorizationCredentials = Depends(security)):
    """Check if Authorization header matches ADMIN_SECRET."""
    if token.credentials != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid Admin Key")
    return True

router = APIRouter(
    prefix="/admin",
    tags=["Admin & Reports"]
)

# Get all attendance records
@router.get("/attendance", response_model=List[Dict[str, Any]])
def get_attendance(admin: bool = Depends(verify_admin)):
    response = supabase.table("attendance_records").select("*").execute()
    return response.data

# Get attendance summary
@router.get("/attendance_summary", response_model=Dict[str, Any])
def get_summary(admin: bool = Depends(verify_admin)):
    response = supabase.table("attendance_records").select("*").execute()
    rows = response.data

    total_present = sum(1 for r in rows if r["status"] == "present")
    total_absent = sum(1 for r in rows if r["status"] == "absent")

    by_student: Dict[str, int] = {}
    for r in rows:
        student = r["students"]
        by_student[student] = by_student.get(student, 0) + (1 if r["status"] == "present" else 0)

    return {
        "total_present": total_present,
        "total_absent": total_absent,
        "by_student": by_student,
    }
