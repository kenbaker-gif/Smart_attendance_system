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

@router.get("/attendance_summary")
def attendance_summary(admin: bool = Depends(verify_admin)):
    try:
        response = supabase.table("attendance_records").select("*").execute()
        rows = response.data or []

        total_present = 0
        total_absent = 0
        by_student = {}

        for r in rows:
            verified = r.get("verified")
            student_id = r.get("student_id", "Unknown")

            if verified == "success":
                total_present += 1
                by_student[student_id] = by_student.get(student_id, 0) + 1
            elif verified == "failed":
                total_absent += 1
            else:
                # Skip or log rows with unexpected verified values
                print(f"[WARN] Unexpected verified value: {verified} for student_id: {student_id}")

        return {
            "total_present": total_present,
            "total_absent": total_absent,
            "by_student": by_student
        }

    except Exception as e:
        print(f"[ERROR] Failed to generate summary: {e}")
        return {"error": str(e)}
