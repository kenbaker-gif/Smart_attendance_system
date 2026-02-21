import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.get("/")
def home():
    return {"status": "Smart Attendance API is Active"}

@app.post("/upload-student-face")
async def upload_student(
    student_id: str = Form(...), 
    file: UploadFile = File(...)
):
    try:
        # 1. Prepare File for Supabase
        file_content = await file.read()
        file_path = f"{student_id}/{file.filename}"

        # 2. Upload to Supabase Storage
        # Replace 'student_faces' with your actual bucket name
        storage_response = supabase.storage.from_("student_faces").upload(
            path=file_path,
            file=file_content,
            file_options={"content-type": file.content_type}
        )

        # 3. Get the Public URL for the image
        image_url = supabase.storage.from_("student_faces").get_public_url(file_path)

        # 4. Return the URL to Flutter (Flutter will save to DB)
        # OR you can add the DB insert here to make it a one-stop shop
        return {"image_url": image_url, "file_path": file_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))