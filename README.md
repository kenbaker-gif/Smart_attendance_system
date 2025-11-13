# 📸 Smart Attendance System

A FastAPI-based smart attendance system that uses facial recognition for student attendance tracking.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation Steps
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate`
4. Install dependencies: `pip install fastapi uvicorn python-multipart`
5. Run the server: `uvicorn main:app --reload`

The application will be available at `http://localhost:8000`

---

## 📋 Project Progress

| Week | Status | Milestone |
|------|--------|-----------|
| Week 1 | ✅ Complete | Project setup |
| Week 2 | ✅ Complete | Camera & Image Upload Functionality |

---

## 📌 Week 2 Milestone: Camera and Image Upload

### Overview
Enabled image capture via webcam and image upload functionality for student attendance tracking. Core backend and frontend interactions were implemented and thoroughly tested.

### ✨ Key Achievements

#### 🎥 Webcam Capture
- Users can capture student images directly from their device camera
- Support for front and back camera switching
- Real-time preview of captured images before submission

#### 📁 File Upload
- Manual image upload capability
- Student ID required for image association
- Dynamic backend routing based on student ID

#### 🔄 Workflow Features
- Automatic page reset after successful submission
- Seamless frontend/backend integration
- Support for both webcam and upload inputs simultaneously

---

## 📂 Project Structure

```
smart_attendance_system/
├── main.py                 # FastAPI server & routes
├── static/
│   ├── index.html         # Frontend UI
│   └── js/
│       └── main.js        # Camera & form handling
└── README.md
```

---

## 🔄 User Workflow

1. Open the system in your browser (`http://localhost:8000`)
2. Capture image using webcam **OR** upload an existing file
3. Enter the Student ID
4. Click Submit → Backend processes the file
5. Page automatically resets for the next input

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/attendance/capture/{student_id}` | Submit student image for attendance |
| `GET` | `/` | Serve main frontend page |

---

## 👤 Author
**Ainebyona Abubaker**


**Last Updated:** November 13, 2025