# main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv

from app.routes.attendance import router as attendance_router
from app.routes.admin import router as admin_router

load_dotenv()

app = FastAPI(title="Smart Attendance System")

# Serve frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Homepage
@app.get("/")
def home():
    index_file = os.path.join("static", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Welcome to Smart Attendance System"}

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}

# Include routers
app.include_router(attendance_router)
app.include_router(admin_router)
