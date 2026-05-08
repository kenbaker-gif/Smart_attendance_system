from fastapi import APIRouter, Request
import smtplib
from email.mime.text import MIMEText
import os

notify_router = APIRouter()

ZOHO_USER = os.getenv("ZOHO_USER")       # abubaker@faceattend.app
ZOHO_PASS = os.getenv("ZOHO_PASSWORD")   # your Zoho SMTP password
ADMIN_EMAIL = "abubaker@faceattend.app"

@notify_router.post("/notify-admin")
async def notify_admin(request: Request):
    payload = await request.json()

    # Supabase sends the new row under "record"
    record = payload.get("record", {})
    name = record.get("name", "Unknown")
    email = record.get("email", "Unknown")
    status = record.get("status", "pending")
    institution_id = record.get("id", "N/A")

    if status != "pending":
        return {"message": "Not a pending signup, skipping."}

    subject = f"🔔 New FaceAttend Signup: {name}"
    body = f"""
A new institution just signed up on FaceAttend and needs your approval.

Institution: {name}
Email: {email}
ID: {institution_id}

Approve here: https://faceattend.app/dashboard.html

– FaceAttend Alert
    """.strip()

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = ZOHO_USER
    msg["To"] = ADMIN_EMAIL

    try:
        with smtplib.SMTP_SSL("smtp.zoho.com", 465) as server:
            server.login(ZOHO_USER, ZOHO_PASS)
            server.sendmail(ZOHO_USER, ADMIN_EMAIL, msg.as_string())
    except Exception as e:
        print(f"Email failed: {e}")
        return {"error": str(e)}

    return {"message": "Notification sent"}