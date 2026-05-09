import os
import smtplib
import hmac
import httpx
from email.mime.text import MIMEText
from fastapi import APIRouter, Request, HTTPException

router = APIRouter()

ZOHO_USER = os.getenv("ZOHO_USER")
ZOHO_PASS = os.getenv("ZOHO_PASSWORD")
ADMIN_EMAIL = "abubaker@faceattend.app"
NOTIFY_WEBHOOK_TOKEN = os.getenv("NOTIFY_WEBHOOK_TOKEN")
SMTP_TIMEOUT_SECONDS = int(os.getenv("SMTP_TIMEOUT_SECONDS", "20"))
EMAIL_PROVIDER = os.getenv("EMAIL_PROVIDER", "smtp").lower()  # smtp | zoho_api

# Zoho Mail API configuration (required when EMAIL_PROVIDER=zoho_api)
ZOHO_API_BASE = os.getenv("ZOHO_API_BASE", "https://mail.zoho.com/api")
ZOHO_OAUTH_BASE = os.getenv("ZOHO_OAUTH_BASE", "https://accounts.zoho.com/oauth/v2")
ZOHO_ACCOUNT_ID = os.getenv("ZOHO_ACCOUNT_ID")
ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")

def _verify_webhook_token(request: Request) -> None:
    if not NOTIFY_WEBHOOK_TOKEN:
        raise HTTPException(status_code=503, detail="notify-admin webhook token is not configured")

    provided_token = request.headers.get("x-webhook-token") or ""
    if not hmac.compare_digest(provided_token, NOTIFY_WEBHOOK_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid webhook token")

@router.post("/notify-admin")
async def notify_admin(request: Request):
    # Require a shared webhook token to prevent arbitrary email triggers.
    _verify_webhook_token(request)

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

    if EMAIL_PROVIDER == "zoho_api":
        await _send_via_zoho_api(subject=subject, body=body, to_email=ADMIN_EMAIL)
        return {"message": "Notification sent via Zoho API"}

    if EMAIL_PROVIDER != "smtp":
        raise HTTPException(status_code=503, detail=f"Unsupported EMAIL_PROVIDER '{EMAIL_PROVIDER}'")

    _send_via_smtp(msg=msg, to_email=ADMIN_EMAIL)
    return {"message": "Notification sent via SMTP"}


@router.get("/notify-admin/health")
async def notify_admin_health(request: Request):
    _verify_webhook_token(request)

    if EMAIL_PROVIDER == "smtp":
        return {
            "ok": bool(ZOHO_USER and ZOHO_PASS),
            "provider": "smtp",
            "checks": {
                "zoho_user_configured": bool(ZOHO_USER),
                "zoho_password_configured": bool(ZOHO_PASS),
            }
        }

    if EMAIL_PROVIDER == "zoho_api":
        try:
            token = await _refresh_zoho_access_token()
            return {
                "ok": True,
                "provider": "zoho_api",
                "checks": {
                    "zoho_user_configured": bool(ZOHO_USER),
                    "zoho_account_id_configured": bool(ZOHO_ACCOUNT_ID),
                    "zoho_client_id_configured": bool(ZOHO_CLIENT_ID),
                    "zoho_client_secret_configured": bool(ZOHO_CLIENT_SECRET),
                    "zoho_refresh_token_configured": bool(ZOHO_REFRESH_TOKEN),
                    "oauth_token_refresh_ok": bool(token),
                }
            }
        except HTTPException as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Zoho API healthcheck failed: {exc.detail}"
            )

    raise HTTPException(status_code=503, detail=f"Unsupported EMAIL_PROVIDER '{EMAIL_PROVIDER}'")


async def _refresh_zoho_access_token() -> str:
    if not all([ZOHO_USER, ZOHO_ACCOUNT_ID, ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN]):
        raise HTTPException(
            status_code=503,
            detail="Zoho API credentials are not fully configured"
        )

    timeout = httpx.Timeout(20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        token_resp = await client.post(
            f"{ZOHO_OAUTH_BASE}/token",
            params={
                "refresh_token": ZOHO_REFRESH_TOKEN,
                "client_id": ZOHO_CLIENT_ID,
                "client_secret": ZOHO_CLIENT_SECRET,
                "grant_type": "refresh_token",
            },
        )
        token_data = token_resp.json()
        access_token = token_data.get("access_token")
        if token_resp.status_code != 200 or not access_token:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to refresh Zoho access token: {token_data}"
            )
        return access_token


async def _send_via_zoho_api(subject: str, body: str, to_email: str) -> None:
    access_token = await _refresh_zoho_access_token()
    timeout = httpx.Timeout(20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        send_resp = await client.post(
            f"{ZOHO_API_BASE}/accounts/{ZOHO_ACCOUNT_ID}/messages",
            headers={
                "Authorization": f"Zoho-oauthtoken {access_token}",
                "Content-Type": "application/json",
            },
            json={
                "fromAddress": ZOHO_USER,
                "toAddress": to_email,
                "subject": subject,
                "content": body,
                "mailFormat": "plaintext",
            },
        )
        if send_resp.status_code not in (200, 201):
            raise HTTPException(
                status_code=502,
                detail=f"Zoho API send failed ({send_resp.status_code}): {send_resp.text}"
            )


def _send_via_smtp(msg: MIMEText, to_email: str) -> None:
    if not ZOHO_USER or not ZOHO_PASS:
        raise HTTPException(status_code=503, detail="Email credentials are not configured")

    ssl_error = None
    try:
        with smtplib.SMTP_SSL("smtp.zoho.com", 465, timeout=SMTP_TIMEOUT_SECONDS) as server:
            server.login(ZOHO_USER, ZOHO_PASS)
            server.sendmail(ZOHO_USER, to_email, msg.as_string())
            return
    except Exception as e:
        ssl_error = e
        print(f"Email SSL failed: {e}")

    try:
        with smtplib.SMTP("smtp.zoho.com", 587, timeout=SMTP_TIMEOUT_SECONDS) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(ZOHO_USER, ZOHO_PASS)
            server.sendmail(ZOHO_USER, to_email, msg.as_string())
            return
    except Exception as e:
        print(f"Email STARTTLS failed: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Email delivery failed. SSL error: {ssl_error}; STARTTLS error: {e}"
        )