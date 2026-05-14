import httpx
import os
import uuid
import hmac
import hashlib
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from app.database import supabase

router = APIRouter(prefix="/api/cart", tags=["payments"])

PESAPAL_ENV = os.getenv("PESAPAL_ENV", "sandbox")

if PESAPAL_ENV == "live":
    PESAPAL_BASE = "https://pay.pesapal.com/v3"
else:
    PESAPAL_BASE = "https://cybqa.pesapal.com/pesapalv3"

CONSUMER_KEY = os.getenv("PESAPAL_CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("PESAPAL_CONSUMER_SECRET")
CALLBACK_URL = os.getenv("PESAPAL_CALLBACK_URL", "https://faceattend.app/dashboard")
IPN_URL = os.getenv("PESAPAL_IPN_URL", "https://faceattend.app/api/cart/ipn")

# ── Order store: now uses Supabase table 'pesapal_orders' ────────────────────────────
# Table columns: order_id (text primary key), institution_id (text), plan (text), created_at (timestamptz)


async def get_pesapal_token() -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{PESAPAL_BASE}/api/Auth/RequestToken",
            json={"consumer_key": CONSUMER_KEY, "consumer_secret": CONSUMER_SECRET},
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        data = resp.json()
        if resp.status_code != 200 or "token" not in data:
            raise HTTPException(status_code=500, detail=f"Pesapal auth failed: {data}")
        return data["token"]


async def get_ipn_id(token: str) -> str:
    async with httpx.AsyncClient() as client:
        # Check existing IPNs first
        resp = await client.get(
            f"{PESAPAL_BASE}/api/URLSetup/GetIpnList",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        ipns = resp.json()
        if isinstance(ipns, list) and len(ipns) > 0:
            return ipns[0]["ipn_id"]

        # Register new IPN if none exist
        resp = await client.post(
            f"{PESAPAL_BASE}/api/URLSetup/RegisterIPN",
            json={"url": IPN_URL, "ipn_notification_type": "GET"},
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        data = resp.json()
        return data["ipn_id"]


class CartRequest(BaseModel):
    plan: str  # "standard" or "enterprise"
    email: str
    first_name: str
    last_name: str
    phone: str = ""
    institution_id: str


PLAN_PRICES = {
    "standard":   {"amount": 10.00,  "currency": "USD", "description": "FaceAttend Standard Plan"},
    "enterprise": {"amount": 150.00, "currency": "USD", "description": "FaceAttend Enterprise Plan"},
}


# ── Create cart ────────────────────────────────────────────────────────────

@router.post("/create-cart")
async def create_cart(payload: CartRequest):
    plan = PLAN_PRICES.get(payload.plan.lower())
    if not plan:
        raise HTTPException(status_code=400, detail="Invalid plan. Choose 'standard' or 'enterprise'.")

    token  = await get_pesapal_token()
    ipn_id = await get_ipn_id(token)
    order_id = str(uuid.uuid4())

    # Store mapping in database so IPN handler can look up institution
    supabase.table("pesapal_orders").insert({
        "order_id": order_id,
        "institution_id": payload.institution_id,
        "plan": payload.plan.lower(),
    }).execute()

    order_payload = {
        "id":           order_id,
        "currency":     plan["currency"],
        "amount":       plan["amount"],
        "description":  plan["description"],
        "callback_url": CALLBACK_URL,
        "notification_id": ipn_id,
        "billing_address": {
            "email_address": payload.email,
            "first_name":    payload.first_name,
            "last_name":     payload.last_name,
            "phone_number":  payload.phone,
        },
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{PESAPAL_BASE}/api/Transactions/SubmitOrderRequest",
            json=order_payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept":        "application/json",
                "Content-Type":  "application/json",
            },
        )
        data = resp.json()

    if resp.status_code != 200 or "redirect_url" not in data:
        raise HTTPException(status_code=500, detail=f"Pesapal order failed: {data}")

    return {
        "redirect_url":      data["redirect_url"],
        "order_tracking_id": data.get("order_tracking_id"),
        "order_id":          order_id,
    }


# ── IPN handler ────────────────────────────────────────────────────────────

def verify_pesapal_signature(params: dict, signature: str) -> bool:
    """Verify Pesapal IPN signature using HMAC-SHA256"""
    # Pesapal signature is computed over specific parameters in order
    order_tracking_id = params.get("OrderTrackingId") or params.get("orderTrackingId")
    order_merchant_reference = params.get("OrderMerchantReference") or params.get("orderMerchantReference")
    order_status = params.get("OrderStatus") or params.get("orderStatus")
    
    if not all([order_tracking_id, order_merchant_reference, order_status]):
        return False
    
    # Pesapal signature is created by concatenating: tracking_id;merchant_ref;status with CONSUMER_SECRET
    message = f"{order_tracking_id};{order_merchant_reference};{order_status}"
    expected_signature = hmac.new(
        CONSUMER_SECRET.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature or "", expected_signature)

@router.get("/ipn")
async def ipn_handler(request: Request):
    params = dict(request.query_params)
    print("IPN received:", params)

    # Verify the signature
    signature = params.get("pesapal_notification_type")  # Note: adjust if Pesapal uses different param
    # For now, we'll verify using status code. In production, implement full Pesapal signature verification
    
    order_tracking_id  = params.get("OrderTrackingId") or params.get("orderTrackingId")
    merchant_reference = params.get("OrderMerchantReference") or params.get("orderMerchantReference")

    if not order_tracking_id:
        print("IPN params:", params)
        return {"status": "ignored", "reason": "no OrderTrackingId"}

    if not merchant_reference:
        return {"status": "ignored", "reason": "no OrderMerchantReference"}

    # Ignore callbacks for unknown references to reduce spoofing risk.
    order_result = supabase.table("pesapal_orders").select("*").eq("order_id", merchant_reference).execute()
    if not order_result.data:
        print(f"Order {merchant_reference} not found in database")
        return {"status": "ignored", "reason": "unknown merchant reference"}
    order = order_result.data[0]

    # Step 1: Authenticate with Pesapal
    try:
        token = await get_pesapal_token()
    except Exception as e:
        print(f"IPN auth failed: {e}")
        return {"status": "error", "reason": "auth failed"}

    # Step 2: Verify transaction status
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{PESAPAL_BASE}/api/Transactions/GetTransactionStatus",
            params={"orderTrackingId": order_tracking_id},
            headers={
                "Authorization": f"Bearer {token}",
                "Accept":        "application/json",
            },
        )
        status_data = resp.json()

    print("Transaction status:", status_data)

    # payment_status_code: 1=COMPLETED, 0=INVALID, 2=FAILED, 3=REVERSED
    payment_status_code = status_data.get("payment_status_code")
    if payment_status_code != 1:
        print(f"Payment not completed: status_code={payment_status_code}")
        return {"status": "ignored", "payment_status_code": payment_status_code}

    # Ensure Pesapal status payload matches the reference we created.
    status_reference = status_data.get("merchant_reference") or status_data.get("merchantReference")
    if status_reference and status_reference != merchant_reference:
        return {"status": "ignored", "reason": "merchant reference mismatch"}

    institution_id = order["institution_id"]
    plan           = order["plan"]  # "standard" or "enterprise"

    # Step 4: Upgrade institution plan in Supabase
    from .dep import supabase_admin
    try:
        supabase_admin.table("institutions").update({
            "plans": plan
        }).eq("id", institution_id).execute()
        print(f"✅ Institution {institution_id} upgraded to {plan}")

        # Clean up order from database
        supabase.table("pesapal_orders").delete().eq("order_id", merchant_reference).execute()

    except Exception as e:
        print(f"❌ Failed to upgrade institution {institution_id}: {e}")
        return {"status": "error", "reason": str(e)}

    return {"status": "ok", "upgraded": institution_id, "plan": plan}