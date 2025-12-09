# main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from app import dbmodule

# -----------------------
# Load environment variables
# -----------------------
load_dotenv()

# -----------------------
# Initialize FastAPI
# -----------------------
app = FastAPI(title="Smart Attendance System")

# -----------------------
# Optional: CORS for frontend requests
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501",
    "https://vivacious-charisma-production.up.railway.app/"],  # Replace "*" with your frontend domain in production
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=[""],
)

# -----------------------
# Serve frontend static files
# -----------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------
# Homepage route
# -----------------------
@app.get("/")
def home():
    index_file = os.path.join("static", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Welcome to Smart Attendance System"}

# -----------------------
# Health check route
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------
# Include routers
# -----------------------
from app.routes.attendance import router as attendance_router
from app.routes.admin import router as admin_router

app.include_router(attendance_router, prefix="/attendance")
app.include_router(admin_router, prefix="/admin")

# -----------------------
# Optional: Example route using dbmodule
# -----------------------
@app.get("/debug/all_attendance")
def debug_all_attendance():
    """
    Example route to fetch all attendance records directly from DB
    """
    try:
        records = dbmodule.get_all_attendance()
        # Convert to list of dicts for JSON response
        return [
            {"id": r[0], "student_id": r[1], "status": r[2], "created_at": str(r[3]), "verified": r[4]}
            for r in records
        ]
    except Exception as e:
        return {"error": str(e)}
