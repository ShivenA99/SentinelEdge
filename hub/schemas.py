"""Pydantic models for all SentinelEdge Hub API contracts."""

from pydantic import BaseModel, Field


class FederatedUpdate(BaseModel):
    """Edge device submits this after local training."""

    device_id: str = Field(
        ..., description="Anonymous device identifier, rotated monthly"
    )
    model_version: int = Field(
        ..., description="Global model version the device was running"
    )
    gradient_delta: list[float] = Field(
        ..., description="Model weight differences (DP-noised)"
    )
    feature_importances: list[float] = Field(
        ..., description="Per-feature importance scores (DP-noised)"
    )
    n_samples: int = Field(
        ..., ge=1, description="Number of local training samples used"
    )
    dp_epsilon: float = Field(
        ..., gt=0, description="Privacy budget spent"
    )
    dp_sigma: float = Field(
        ..., gt=0, description="Noise standard deviation applied"
    )


class AggregationResponse(BaseModel):
    """Hub returns this after aggregation round."""

    model_version: int
    model_delta: str = Field(
        ..., description="Base64-encoded compressed weight delta"
    )
    signature: str = Field(
        ..., description="Ed25519 signature (hex)"
    )
    n_contributing_devices: int
    round_accuracy: float


class ModelVersionInfo(BaseModel):
    """Lightweight model version metadata."""

    model_version: int
    created_at: str
    n_contributing_devices: int
    round_accuracy: float


class GlobalMetrics(BaseModel):
    """Aggregated global metrics -- no per-device data exposed."""

    total_rounds: int
    current_model_version: int
    current_accuracy: float
    total_contributing_devices: int
    accuracy_history: list[float]


class RoundStatus(BaseModel):
    """Status of the current federated learning round."""

    round_id: int
    status: str = Field(
        ..., description='One of: "collecting", "aggregating", "complete"'
    )
    updates_received: int
    min_updates_required: int
    model_version: int
