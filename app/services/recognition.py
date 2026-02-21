"""
Face Recognition Service
Handles InsightFace model operations and encoding management

HOW IT WORKS:
=============

1. INITIALIZATION:
   - Loads InsightFace buffalo_s model (optimized for CPU)
   - Loads face encodings from local cache or Supabase
   - Ready to verify faces immediately

2. VERIFICATION FLOW:
   - Receives image from Flask/API
   - Detects faces using InsightFace
   - Extracts face embeddings
   - Compares against known encodings
   - Returns match result with confidence score

3. ENCODING GENERATION:
   - Admin triggers sync via /admin/sync-encodings
   - Downloads all student images from Supabase
   - Extracts face encodings from each image
   - Saves locally and uploads to cloud
   - Takes ~1-2 min for 100+ students

PERFORMANCE NOTES:
- First request takes longer (model loading)
- Subsequent requests are fast (<100ms)
- Memory: ~500MB-1GB during operation
- CPU: Single core is sufficient
"""

import os
import gc
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import cv2
from app.utils.logger import logger
from app.models import AttendanceRecord
from datetime import datetime, timedelta

class RecognitionService:
    """Service for face recognition operations"""
    
    def __init__(self, data_dir: Path, model_name: str = "buffalo_s", threshold: float = 0.50):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.threshold = threshold
        self.encodings_path = self.data_dir / "encodings_insightface.pkl"
        
        self._app = None  # InsightFace engine
        self._encodings: Optional[np.ndarray] = None
        self._ids: Optional[list] = None
        self._initialize_engine()
        self._load_encodings()
    
    def _initialize_engine(self):
        """Initialize InsightFace face analysis engine"""
        try:
            from insightface.app import FaceAnalysis
            logger.info(f"Initializing InsightFace ({self.model_name})...")
            self._app = FaceAnalysis(name=self.model_name, providers=["CPUExecutionProvider"])
            self._app.prepare(ctx_id=-1, det_size=(640, 640))
            logger.info("✅ InsightFace initialized")
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            self._app = None
    
    def is_initialized(self) -> bool:
        """Check if service is fully initialized"""
        return self._app is not None
    
    def _load_encodings(self):
        """Load face encodings from local cache or Supabase"""
        if self.encodings_path.exists():
            try:
                with open(self.encodings_path, "rb") as f:
                    data = pickle.load(f)
                self._encodings = self._normalize_encodings(np.array(data["encodings"]))
                self._ids = [str(i) for i in data["ids"]]
                logger.info(f"✅ Loaded {len(self._ids)} encodings from cache")
                return
            except Exception as e:
                logger.warning(f"Failed to load local encodings: {e}")
        
        # Try to download from Supabase if available
        if os.getenv("USE_SUPABASE", "false").lower() == "true":
            self._load_encodings_from_supabase()
        else:
            logger.warning("No encodings found and Supabase not configured")
            self._encodings = np.array([])
            self._ids = []
    
    def _load_encodings_from_supabase(self):
        """Download encodings from Supabase cloud storage"""
        try:
            from supabase import create_client
            
            supabase_url = os.getenv("SUPABASE_URL", "")
            supabase_key = os.getenv("SUPABASE_KEY", "")
            supabase_bucket = os.getenv("SUPABASE_BUCKET", "")
            remote_path = os.getenv("ENCODINGS_REMOTE_PATH", "encodings/encodings_insightface.pkl")
            
            if not all([supabase_url, supabase_key, supabase_bucket]):
                logger.warning("Supabase credentials incomplete")
                return
            
            supabase = create_client(supabase_url, supabase_key)
            logger.info("Downloading encodings from Supabase...")
            
            res = supabase.storage.from_(supabase_bucket).download(remote_path.lstrip('/'))
            data_bytes = res if isinstance(res, (bytes, bytearray)) else getattr(res, 'content', None)
            
            if data_bytes:
                with open(self.encodings_path, "wb") as f:
                    f.write(data_bytes)
                
                data = pickle.loads(data_bytes)
                self._encodings = self._normalize_encodings(np.array(data["encodings"]))
                self._ids = [str(i) for i in data["ids"]]
                logger.info(f"✅ Loaded {len(self._ids)} encodings from Supabase")
        except Exception as e:
            logger.error(f"Failed to load from Supabase: {e}")
    
    @staticmethod
    def _normalize_encodings(vectors: np.ndarray) -> np.ndarray:
        """Normalize embedding vectors"""
        if vectors.size == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms
    
    def generate_encodings(self, supabase=None) -> Dict[str, int]:
        """Generate encodings from all student images in Supabase"""
        if not self._app:
            raise RuntimeError("Recognition engine not initialized")
        
        if not supabase:
            raise ValueError("Supabase client required for encoding generation")
        
        supabase_bucket = os.getenv("SUPABASE_BUCKET", "")
        
        logger.info("Starting encoding generation from Supabase...")
        
        try:
            # Get all student folders
            folders = supabase.storage.from_(supabase_bucket).list()
            valid_folders = [f for f in folders if not f['name'].startswith('.') and f['name'] != "encodings"]
            
            if not valid_folders:
                raise ValueError("No student folders found")
            
            encodings_list = []
            ids_list = []
            processed_count = 0
            
            for folder in valid_folders:
                student_id = folder['name']
                
                try:
                    files = supabase.storage.from_(supabase_bucket).list(student_id)
                    
                    for f_info in files:
                        file_name = f_info['name']
                        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            remote_path = f"{student_id}/{file_name}"
                            
                            try:
                                data = supabase.storage.from_(supabase_bucket).download(remote_path)
                                nparr = np.frombuffer(data, np.uint8)
                                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                
                                if img_bgr is None:
                                    continue
                                
                                # Detect faces
                                faces = self._app.get(img_bgr)
                                if faces:
                                    # Use largest face
                                    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                                    encodings_list.append(face.embedding)
                                    ids_list.append(student_id)
                                    processed_count += 1
                            except Exception as e:
                                logger.warning(f"Error processing {remote_path}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing student {student_id}: {e}")
            
            if not encodings_list:
                raise ValueError("No faces detected in images")
            
            # Save locally and to Supabase
            arr = self._normalize_encodings(np.array(encodings_list, dtype=np.float32))
            
            with open(self.encodings_path, "wb") as f:
                pickle.dump({"encodings": arr, "ids": np.array(ids_list)}, f)
            
            # Upload to Supabase
            encodings_remote_path = os.getenv("ENCODINGS_REMOTE_PATH", "encodings/encodings_insightface.pkl")
            with open(self.encodings_path, "rb") as f:
                supabase.storage.from_(supabase_bucket).upload(
                    path=encodings_remote_path.lstrip('/'),
                    file=f,
                    file_options={"upsert": "true"}
                )
            
            self._encodings = arr
            self._ids = ids_list
            
            logger.info(f"✅ Generated {processed_count} encodings from {len(valid_folders)} students")
            gc.collect()
            
            return {
                "encodings_count": processed_count,
                "students_count": len(valid_folders)
            }
        
        except Exception as e:
            logger.error(f"Encoding generation failed: {e}")
            raise
    
    def sync_encodings(self) -> Dict[str, int]:
        """Sync encodings from Supabase"""
        try:
            from supabase import create_client
            
            supabase_url = os.getenv("SUPABASE_URL", "")
            supabase_key = os.getenv("SUPABASE_KEY", "")
            
            if not all([supabase_url, supabase_key]):
                raise ValueError("Supabase credentials not configured")
            
            supabase = create_client(supabase_url, supabase_key)
            return self.generate_encodings(supabase)
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            raise
    
    def verify_identity(self, img_bgr: np.ndarray, student_id: str, db) -> Dict:
        """
        Verify a student's identity against the face database
        
        Returns dict with keys: success (bool), message (str), confidence (float)
        """
        if not self._app:
            return {
                "success": False,
                "message": "Recognition service not initialized",
                "confidence": 0.0
            }
        
        if self._encodings is None or self._encodings.size == 0:
            return {
                "success": False,
                "message": "No encodings available in database",
                "confidence": 0.0
            }
        
        try:
            # Detect face in image
            faces = self._app.get(img_bgr)
            
            if not faces:
                return {
                    "success": False,
                    "message": "No face detected in image",
                    "confidence": 0.0
                }
            
            # Use the largest face
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            captured_emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-10)
            
            # Compare against known encodings
            dists = 1.0 - np.dot(self._encodings, captured_emb)
            idx = np.argmin(dists)
            confidence = float(1.0 - dists[idx])
            matched_id = self._ids[idx]
            
            # Verify match
            if dists[idx] < (1.0 - self.threshold) and str(matched_id) == str(student_id).strip():
                return {
                    "success": True,
                    "message": f"Identity verified successfully",
                    "confidence": confidence
                }
            else:
                return {
                    "success": False,
                    "message": f"Identity mismatch or low confidence",
                    "confidence": confidence
                }
        
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return {
                "success": False,
                "message": f"Verification failed: {str(e)}",
                "confidence": 0.0
            }
