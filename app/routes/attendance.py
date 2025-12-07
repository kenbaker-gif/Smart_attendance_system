# app/routes/attendance.py
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import RedirectResponse
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

router = APIRouter(
    prefix="",  # No /admin prefix
    tags=["Attendance"]
)

@router.post("/capture/{student_id}")
async def capture_and_upload(student_id: str, file: UploadFile = File(...)):
    """
    Capture and upload student attendance.
    """
    # Example: upload file to Supabase bucket
    content = await file.read()
    filename = f"{student_id}/{file.filename}"

    supabase.storage.from_(SUPABASE_BUCKET).upload(filename, content)
    
    return RedirectResponse(url="/", status_code=303)
