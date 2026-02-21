"""
FastAPI Backend for Smart Attendance System
Provides REST endpoints for face recognition and attendance tracking

DEPLOYMENT GUIDE:
================

LOCAL DEVELOPMENT:
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

RAILWAY DEPLOYMENT:
  1. railway login
  2. railway init
  3. railway up
  4. railway status  # Get your public URL

FLUTTER APP INTEGRATION:
  Replace 'apiBaseUrl' in Flutter main.dart:
  const apiBaseUrl = 'https://your-railway-app-url.railway.app';
  
  Then call endpoints:
  - POST /verify (with image file)
  - GET /attendance-records
  - GET /students
  - POST /admin/sync-encodings (with Authorization header)

API ENDPOINTS:
==============
GET  /              - API info
GET  /health        - Health check
POST /verify        - Verify student identity (requires: student_id, image file)
GET  /attendance-records - Get attendance logs (optional: student_id, limit)
GET  /students      - List all students
POST /admin/sync-encodings - Regenerate encodings (requires: Authorization header)

ENVIRONMENT VARIABLES (set in Railway dashboard):
==================================================
PORT=8000
THRESHOLD=0.50
USE_SUPABASE=true
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-key
SUPABASE_BUCKET=attendance-images
DB_USER=postgres
DB_PASSWORD=your-password
DB_HOST=your-host.supabase.com
DB_PORT=5432
DB_NAME=postgres
ADMIN_SECRET=your-secret-key
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager
import logging
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from app.database import get_db, SessionLocal
from app.models import Student, AttendanceRecord
from app.services.recognition import RecognitionService
from app.utils.logger import logger

# -=- Configuration -=-
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "attendance.log"

for d in [DATA_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Configuration from environment
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.50"))
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")
USE_SUPABASE = os.getenv("USE_SUPABASE", "false").lower() == "true"

# -=- Setup Logging -=-
if not logger.handlers:
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

# -=- Global Recognition Service -=-
recognition_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events for FastAPI"""
    global recognition_service
    logger.info("Starting Smart Attendance System API")
    
    # Initialize recognition service on startup
    recognition_service = RecognitionService(
        data_dir=DATA_DIR,
        model_name="buffalo_s",
        threshold=DEFAULT_THRESHOLD
    )
    
    yield
    
    logger.info("Shutting down Smart Attendance System API")

# -=- Create FastAPI App -=-
app = FastAPI(
    title="Smart Attendance System API",
    description="Face recognition and attendance tracking API",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -=- Request/Response Models -=-
class VerificationRequest(BaseModel):
    student_id: str
    
class VerificationResponse(BaseModel):
    success: bool
    message: str
    student_id: str | None = None
    confidence: float | None = None

class AttendanceRecordResponse(BaseModel):
    id: int
    student_id: str
    confidence: float
    detection_method: str
    verified: str
    timestamp: str

class SyncResponse(BaseModel):
    success: bool
    message: str
    encodings_count: int = 0
    students_count: int = 0

class HealthResponse(BaseModel):
    status: str
    service_ready: bool

# -=- Root Endpoint -=-
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Smart Attendance System API",
        "version": "1.0.0",
        "description": "Face recognition and attendance tracking REST API",
        "docs": "http://localhost:8000/docs",
        "endpoints": {
            "health": "/health",
            "verify": "/verify (POST)",
            "attendance_records": "/attendance-records (GET)",
            "students": "/students (GET)",
            "admin_sync": "/admin/sync-encodings (POST)"
        }
    }

# -=- Health Check -=-
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service_ready": recognition_service is not None and recognition_service.is_initialized()
    }

# -=- Face Verification Endpoint -=-
@app.post("/verify", response_model=VerificationResponse)
async def verify_face(
    student_id: str,
    file: UploadFile = File(...),
    db = Depends(get_db)
):
    """
    Verify a student's identity using face recognition
    
    Args:
        student_id: Student ID to verify against
        file: Image file containing the face to verify
    """
    try:
        if not recognition_service or not recognition_service.is_initialized():
            raise HTTPException(status_code=503, detail="Recognition service not initialized")
        
        # Read image from upload
        image_data = await file.read()
        img = Image.open(BytesIO(image_data)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Perform verification
        result = recognition_service.verify_identity(
            img_bgr,
            student_id,
            db
        )
        
        # Log to database
        if result["success"]:
            attendance = AttendanceRecord(
                student_id=student_id,
                confidence=result["confidence"],
                detection_method="insightface_buffalo_s",
                verified="success"
            )
            logger.info(f"Attendance verified: {student_id} (confidence: {result['confidence']:.2f})")
        else:
            attendance = AttendanceRecord(
                student_id=student_id,
                confidence=result.get("confidence", 0.0),
                detection_method="insightface_buffalo_s",
                verified="failed"
            )
            logger.warning(f"Verification failed for {student_id}")
        
        db.add(attendance)
        db.commit()
        
        return VerificationResponse(
            success=result["success"],
            message=result["message"],
            student_id=student_id if result["success"] else None,
            confidence=result.get("confidence")
        )
        
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if db:
            db.close()

# -=- Get Attendance Records -=-
@app.get("/attendance-records", response_model=list[AttendanceRecordResponse])
async def get_attendance_records(
    student_id: str | None = None,
    limit: int = 100,
    db = Depends(get_db)
):
    """
    Get attendance records, optionally filtered by student_id
    """
    try:
        query = db.query(AttendanceRecord)
        
        if student_id:
            query = query.filter(AttendanceRecord.student_id == student_id)
        
        records = query.order_by(AttendanceRecord.timestamp.desc()).limit(limit).all()
        
        return [
            AttendanceRecordResponse(
                id=r.id,
                student_id=r.student_id,
                confidence=r.confidence,
                detection_method=r.detection_method,
                verified=r.verified,
                timestamp=r.timestamp.isoformat()
            )
            for r in records
        ]
    except Exception as e:
        logger.error(f"Error fetching records: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if db:
            db.close()

# -=- Admin Endpoints -=-
@app.post("/admin/sync-encodings", response_model=SyncResponse)
async def sync_encodings(
    authorization: str = Header(None),
    db = Depends(get_db)
):
    """
    Admin endpoint to regenerate face encodings from Supabase
    Requires ADMIN_SECRET in Authorization header
    """
    try:
        # Validate admin secret
        if authorization != f"Bearer {ADMIN_SECRET}":
            logger.warning("Unauthorized sync attempt")
            raise HTTPException(status_code=401, detail="Invalid admin secret")
        
        if not recognition_service or not USE_SUPABASE:
            raise HTTPException(status_code=400, detail="Recognition service or Supabase not configured")
        
        logger.info("Starting encoding synchronization")
        stats = recognition_service.sync_encodings()
        
        logger.info(f"Sync complete: {stats['encodings_count']} encodings, {stats['students_count']} students")
        
        return SyncResponse(
            success=True,
            message="Encodings synchronized successfully",
            **stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sync error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if db:
            db.close()

# -=- Get Students -=-
@app.get("/students")
async def get_students(limit: int = 100, db = Depends(get_db)):
    """Get list of all students"""
    try:
        students = db.query(Student).limit(limit).all()
        return [
            {
                "id": s.id,
                "name": s.name,
                "email": s.email,
                "created_at": s.created_at.isoformat() if s.created_at else None
            }
            for s in students
        ]
    except Exception as e:
        logger.error(f"Error fetching students: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if db:
            db.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENVIRONMENT", "production") != "production"
    )
