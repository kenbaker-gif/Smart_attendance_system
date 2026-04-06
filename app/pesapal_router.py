import httpx
import os
import uuid
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

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
    "standard": {"amount": 49.00, "currency": "USD", "description": "FaceAttend Standard Plan"},
    "enterprise": {"amount": 149.00, "currency": "USD", "description": "FaceAttend Enterprise Plan"},
}


@router.post("/create-cart")
async def create_cart(payload: CartRequest):
    plan = PLAN_PRICES.get(payload.plan.lower())
    if not plan:
        raise HTTPException(status_code=400, detail="Invalid plan. Choose 'standard' or 'enterprise'.")

    token = await get_pesapal_token()
    ipn_id = await get_ipn_id(token)
    order_id = str(uuid.uuid4())

    order_payload = {
        "id": order_id,
        "currency": plan["currency"],
        "amount": plan["amount"],
        "description": plan["description"],
        "callback_url": CALLBACK_URL,
        "notification_id": ipn_id,
        "billing_address": {
            "email_address": payload.email,
            "first_name": payload.first_name,
            "last_name": payload.last_name,
            "phone_number": payload.phone,
        },
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{PESAPAL_BASE}/api/Transactions/SubmitOrderRequest",
            json=order_payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        data = resp.json()

    if resp.status_code != 200 or "redirect_url" not in data:
        raise HTTPException(status_code=500, detail=f"Pesapal order failed: {data}")

    return {
        "redirect_url": data["redirect_url"],
        "order_tracking_id": data.get("order_tracking_id"),
        "order_id": order_id,
    }


@router.get("/ipn")
async def ipn_handler(request: Request):
    params = dict(request.query_params)
    # TODO: verify transaction status and update institution plan in Supabase
    print("IPN received:", params)
    return {"status": "ok"}