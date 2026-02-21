import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from supabase import create_client
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")
BUCKET = os.getenv("SUPABASE_BUCKET", "student_faces")

if URL and not URL.endswith("/"):
    URL += "/"

supabase = create_client(URL, KEY)
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Smart Attendance API is running", "status": "online"}

@app.post("/upload-student-face")
async def upload_student_face(
    student_id: str = Form(...), 
    file: UploadFile = File(...)
):
    try:
        # 1. Determine the next sequence number (listing files in the folder)
        res = supabase.storage.from_(BUCKET).list(student_id)
        # Handle cases where res might be empty or None
        existing_files = [f['name'] for f in (res or []) if f['name'].lower().endswith(('.jpg', '.jpeg', '.png'))]
        next_num = len(existing_files) + 1
        
        # 2. Read file content
        file_content = await file.read()
        
        # 3. Upload Loop (Collision Protection)
        uploaded = False
        while not uploaded:
            file_name = f"{next_num}.jpg"
            path = f"{student_id}/{file_name}"
            try:
                supabase.storage.from_(BUCKET).upload(
                    path=path,
                    file=file_content,
                    file_options={"content-type": "image/jpeg"}
                )
                uploaded = True
            except Exception as e:
                if "409" in str(e) or "already exists" in str(e).lower():
                    next_num += 1
                else:
                    raise e

        return {
            "status": "success",
            "message": f"Successfully saved as {file_name}",
            "student_id": student_id,
            "file_path": path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))