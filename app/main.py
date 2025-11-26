from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes import attendance
import os
from fastapi.responses import FileResponse

app = FastAPI(title="Smart Attendance System")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Homepage
@app.get("/")
def home():
    return FileResponse(os.path.join("static", "index.html"))

# Healthcheck endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Include router
app.include_router(attendance.router, prefix="/attendance", tags=["Attendance"])