import os
import smtplib
import hmac
from email.mime.text import MIMEText
from fastapi import APIRouter, Request, HTTPException

router = APIRouter()

ZOHO_USER = os.getenv("ZOHO_USER")
ZOHO_PASS = os.getenv("ZOHO_PASSWORD")
ADMIN_EMAIL = "abubaker@faceattend.app"
NOTIFY_WEBHOOK_TOKEN = os.getenv("NOTIFY_WEBHOOK_TOKEN")
SMTP_TIMEOUT_SECONDS = int(os.getenv("SMTP_TIMEOUT_SECONDS", "20"))

@router.post("/notify-admin")
async def notify_admin(request: Request):
    # Require a shared webhook token to prevent arbitrary email triggers.
    if not NOTIFY_WEBHOOK_TOKEN:
        raise HTTPException(status_code=503, detail="notify-admin webhook token is not configured")

    provided_token = request.headers.get("x-webhook-token") or ""
    if not hmac.compare_digest(provided_token, NOTIFY_WEBHOOK_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid webhook token")

    payload = await request.json()

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

Approve here: https://faceattend.app/dashboard

– FaceAttend Alert
    """.strip()

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = ZOHO_USER
    msg["To"] = ADMIN_EMAIL

    if not ZOHO_USER or not ZOHO_PASS:
        raise HTTPException(status_code=503, detail="Email credentials are not configured")

    ssl_error = None
    try:
        with smtplib.SMTP_SSL("smtp.zoho.com", 465, timeout=SMTP_TIMEOUT_SECONDS) as server:
            server.login(ZOHO_USER, ZOHO_PASS)
            server.sendmail(ZOHO_USER, ADMIN_EMAIL, msg.as_string())
            return {"message": "Notification sent via SSL"}
    except Exception as e:
        ssl_error = e
        print(f"Email SSL failed: {e}")

    try:
        with smtplib.SMTP("smtp.zoho.com", 587, timeout=SMTP_TIMEOUT_SECONDS) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(ZOHO_USER, ZOHO_PASS)
            server.sendmail(ZOHO_USER, ADMIN_EMAIL, msg.as_string())
            return {"message": "Notification sent via STARTTLS"}
    except Exception as e:
        print(f"Email STARTTLS failed: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Email delivery failed. SSL error: {ssl_error}; STARTTLS error: {e}"
        )

    return {"message": "Notification sent"}