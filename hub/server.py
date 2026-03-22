"""SentinelEdge Hub - Federated Aggregation Server.

A lightweight coordination server that:
- Accepts DP-noised gradient deltas from edge devices
- Performs federated averaging with Byzantine fault detection
- Validates aggregated models against a held-out set
- Signs and distributes improved global models

Run with:
    python -m hub.server
    uvicorn hub.server:app --host 0.0.0.0 --port 8080
"""

import logging
import time
import uuid
from collections import defaultdict

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model_store import ModelStore
from .round_manager import RoundManager
from .schemas import (
    AggregationResponse,
    DeviceRegistration,
    DeviceRegistrationResponse,
    FederatedUpdate,
    GlobalMetrics,
    ModelVersionInfo,
    RoundStatus,
)
from .validator import ModelValidator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SentinelEdge Hub",
    description=(
        "Federated aggregation server for SentinelEdge -- "
        "coordinates edge devices performing on-device phone call fraud detection."
    ),
    version="0.1.0",
)

# CORS for frontend / dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Component initialisation
# ---------------------------------------------------------------------------
model_store = ModelStore()
validator = ModelValidator()
round_manager = RoundManager(
    min_devices=5,
    model_store=model_store,
    validator=validator,
)

# ---------------------------------------------------------------------------
# Authentication state (in-memory)
# ---------------------------------------------------------------------------

# api_key -> device_id
_api_keys: dict[str, str] = {}

# Rate limiting: api_key -> list of submission timestamps
_RATE_LIMIT_MAX = 10  # max submissions per window
_RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds
_submission_timestamps: dict[str, list[float]] = defaultdict(list)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


async def verify_api_key(x_api_key: str = Header(...)) -> tuple[str, str]:
    """Validate the X-API-Key header.

    Returns a ``(api_key, device_id)`` tuple so that downstream handlers
    can use the key for rate-limit tracking without a reverse lookup.

    Raises 401 if the key is not recognised.
    """
    if x_api_key not in _api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key, _api_keys[x_api_key]


def _check_rate_limit(api_key: str) -> None:
    """Enforce per-key submission rate limit.

    Raises 429 if the device has exceeded the maximum number of
    submissions within the rolling time window.
    """
    now = time.time()
    cutoff = now - _RATE_LIMIT_WINDOW

    # Prune old timestamps
    _submission_timestamps[api_key] = [
        ts for ts in _submission_timestamps[api_key] if ts > cutoff
    ]

    if len(_submission_timestamps[api_key]) >= _RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: max {_RATE_LIMIT_MAX} submissions "
                f"per {_RATE_LIMIT_WINDOW // 60} minutes"
            ),
        )

    _submission_timestamps[api_key].append(now)


# ---------------------------------------------------------------------------
# Public endpoints (no auth required)
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "sentineledge-hub"}


@app.get("/v1/model/version", response_model=ModelVersionInfo)
async def get_model_version():
    """Check current model version -- edge devices poll this to decide
    whether to download a newer model.
    """
    version, _ = model_store.get_latest()
    if version == 0:
        raise HTTPException(status_code=404, detail="No model available yet")

    meta = model_store.get_metadata(version)
    if meta is None:
        raise HTTPException(status_code=404, detail="Model metadata not found")

    return ModelVersionInfo(
        model_version=version,
        created_at=meta.get("created_at", ""),
        n_contributing_devices=meta.get("n_contributing_devices", 0),
        round_accuracy=meta.get("accuracy", 0.0),
    )


@app.get("/v1/round/status", response_model=RoundStatus)
async def get_round_status():
    """Get current federated round status."""
    return round_manager.get_status()


@app.get("/v1/metrics/global", response_model=GlobalMetrics)
async def get_global_metrics():
    """Aggregated stats only: total rounds, accuracy trend, etc.

    No per-device data is exposed -- privacy by design.
    """
    metrics = round_manager.get_global_metrics()
    return GlobalMetrics(**metrics)


@app.get("/v1/model/public_key")
async def get_public_key():
    """Return the hub's Ed25519 public key (hex) for signature verification.

    Edge devices use this to verify that model updates are authentic.
    """
    return {"public_key": model_store.get_public_key()}


@app.post(
    "/v1/devices/register",
    response_model=DeviceRegistrationResponse,
)
async def register_device(registration: DeviceRegistration):
    """Register a new edge device and receive an API key.

    The API key must be included in the ``X-API-Key`` header for all
    authenticated endpoints (federated submit, model download).
    """
    api_key = str(uuid.uuid4())
    _api_keys[api_key] = registration.device_id
    logger.info(
        "Device registered: device_id=%s (api_key=%s...)",
        registration.device_id,
        api_key[:8],
    )
    return DeviceRegistrationResponse(
        api_key=api_key,
        device_id=registration.device_id,
    )


# ---------------------------------------------------------------------------
# Authenticated endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/federated/submit", response_model=RoundStatus)
async def submit_update(
    update: FederatedUpdate,
    auth: tuple[str, str] = Depends(verify_api_key),
):
    """Edge device submits a DP-noised gradient delta.

    Requires a valid ``X-API-Key`` header.

    The hub collects updates until `min_devices` are reached, then
    automatically triggers federated averaging, validation, and
    (if accepted) publishes a new global model version.
    """
    api_key, device_id = auth
    _check_rate_limit(api_key)

    try:
        status = await round_manager.submit_update(update)
        return status
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Error processing update: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal aggregation error")


@app.get("/v1/model/latest", response_model=AggregationResponse)
async def get_latest_model(
    auth: tuple[str, str] = Depends(verify_api_key),
):
    """Edge device downloads the latest global model as a delta patch.

    Requires a valid ``X-API-Key`` header.

    Returns a base64-encoded gzip-compressed weight delta, along with
    the Ed25519 signature so the device can verify authenticity.
    """
    version, weights = model_store.get_latest()
    if version == 0:
        raise HTTPException(status_code=404, detail="No model available yet")

    try:
        # Create delta from version 0 (full weights) -- edge devices
        # that already have a previous version can request a specific delta
        # via /v1/model/delta/{old}/{new} (future endpoint).
        model_delta_b64 = model_store.create_delta_patch_b64(
            old_version=0, new_version=version
        )
        signature = model_store.get_signature_for_version(version)
        meta = model_store.get_metadata(version)

        return AggregationResponse(
            model_version=version,
            model_delta=model_delta_b64,
            signature=signature,
            n_contributing_devices=meta.get("n_contributing_devices", 0) if meta else 0,
            round_accuracy=meta.get("accuracy", 0.0) if meta else 0.0,
        )
    except Exception as exc:
        logger.error("Error creating model response: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to prepare model")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
