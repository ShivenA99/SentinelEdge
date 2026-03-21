"""Federated learning round manager.

Manages the lifecycle of federated learning rounds:
1. Collect gradient updates from edge devices (in memory only)
2. Once min_devices updates arrive, trigger aggregation
3. Validate the aggregated model
4. If accepted, publish the new global model version

All state is kept in memory -- nothing persisted to disk beyond model
weights (handled by ModelStore).
"""

import asyncio
import logging
import time
from datetime import datetime, timezone

import numpy as np

from .aggregator import FedAvgAggregator
from .model_store import ModelStore
from .schemas import FederatedUpdate, RoundStatus
from .validator import ModelValidator

logger = logging.getLogger(__name__)


class RoundManager:
    """Manage federated learning rounds -- collect K updates, trigger aggregation."""

    def __init__(
        self,
        min_devices: int = 5,
        round_timeout: int = 86400,
        model_store: ModelStore | None = None,
        validator: ModelValidator | None = None,
        n_features: int = 25,
    ):
        """Initialize the round manager.

        Args:
            min_devices: Minimum updates required before triggering aggregation.
            round_timeout: Maximum seconds to wait for updates before discarding.
            model_store: ModelStore instance for persisting models.
            validator: ModelValidator instance for validating aggregated models.
            n_features: Number of model features (for initial model).
        """
        self.min_devices = min_devices
        self.round_timeout = round_timeout
        self.n_features = n_features

        self.current_round: int = 0
        self.updates: list[FederatedUpdate] = []
        self.round_start_time: float = time.time()
        self.status: str = "collecting"  # "collecting" | "aggregating" | "complete"

        self.aggregator = FedAvgAggregator()
        self.model_store = model_store or ModelStore()
        self.validator = validator or ModelValidator(n_features=n_features)

        # Global tracking
        self.round_history: list[dict] = []
        self.total_contributing_devices: int = 0
        self.accuracy_history: list[float] = []
        self.current_accuracy: float = 0.0

        # Lock to prevent concurrent aggregation
        self._aggregation_lock = asyncio.Lock()
        # Track device IDs that already submitted this round
        self._submitted_devices: set[str] = set()

        # Initialize with a baseline model if none exists
        self._ensure_initial_model()

    def _ensure_initial_model(self):
        """Create initial model v1 if store is empty."""
        version, weights = self.model_store.get_latest()
        if version == 0:
            # Create a small random initial model
            rng = np.random.RandomState(0)
            initial_weights = rng.randn(self.n_features) * 0.01
            metrics = self.validator.compute_metrics(initial_weights)
            initial_accuracy = metrics["f1"]

            self.model_store.save_model(
                weights=initial_weights,
                version=1,
                accuracy=initial_accuracy,
                n_contributing_devices=0,
            )
            self.current_accuracy = initial_accuracy
            self.accuracy_history.append(initial_accuracy)
            logger.info(
                "Initialized baseline model v1 (F1=%.4f)", initial_accuracy
            )
        else:
            meta = self.model_store.get_metadata(version)
            if meta:
                self.current_accuracy = meta.get("accuracy", 0.0)
                self.accuracy_history.append(self.current_accuracy)
            logger.info("Existing model v%d loaded", version)

    async def submit_update(self, update: FederatedUpdate) -> RoundStatus:
        """Accept an update from an edge device.

        If min_devices reached, trigger aggregation automatically.
        Updates are stored in memory only -- never persisted to disk.

        Duplicate submissions from the same device_id within a round
        are rejected.

        Args:
            update: Federated update from an edge device.

        Returns:
            Current round status.

        Raises:
            ValueError: If the update is invalid or a duplicate.
        """
        # Reject if currently aggregating
        if self.status == "aggregating":
            raise ValueError(
                "Aggregation in progress, please retry after completion"
            )

        # Check for stale round (timeout)
        if time.time() - self.round_start_time > self.round_timeout:
            logger.warning(
                "Round %d timed out after %ds, resetting",
                self.current_round,
                self.round_timeout,
            )
            self._reset_round()

        # Reject duplicate device submissions within the same round
        if update.device_id in self._submitted_devices:
            raise ValueError(
                f"Device {update.device_id} already submitted for round {self.current_round}"
            )

        # Basic validation
        current_version, _ = self.model_store.get_latest()
        if update.model_version > current_version:
            raise ValueError(
                f"Update references model v{update.model_version} "
                f"but latest is v{current_version}"
            )

        # Accept the update
        self.updates.append(update)
        self._submitted_devices.add(update.device_id)

        logger.info(
            "Round %d: received update from %s (v%d, %d samples, epsilon=%.2f) "
            "[%d/%d]",
            self.current_round,
            update.device_id,
            update.model_version,
            update.n_samples,
            update.dp_epsilon,
            len(self.updates),
            self.min_devices,
        )

        # Auto-trigger aggregation when threshold reached
        if len(self.updates) >= self.min_devices:
            # Run aggregation in background without blocking the response
            asyncio.create_task(self._run_aggregation())

        return self.get_status()

    async def _run_aggregation(self):
        """Execute aggregation under lock to prevent races."""
        async with self._aggregation_lock:
            if self.status == "aggregating":
                return
            self.status = "aggregating"
            try:
                await self.trigger_aggregation()
            except Exception as e:
                logger.error("Aggregation failed: %s", e, exc_info=True)
                self.status = "collecting"

    async def trigger_aggregation(self) -> np.ndarray:
        """Run FedAvg, validate, and update the global model.

        Steps:
        1. Detect and remove outlier updates (Byzantine tolerance)
        2. Compute weighted average of gradient deltas
        3. Apply delta to current model weights
        4. Validate new model against held-out set
        5. If F1 doesn't drop >2%, accept and publish

        Returns:
            New global model weights.
        """
        logger.info(
            "Round %d: triggering aggregation with %d updates",
            self.current_round,
            len(self.updates),
        )

        updates_to_use = list(self.updates)
        n_devices = len(updates_to_use)

        # Step 1 & 2: Aggregate with outlier detection
        global_delta, feature_importances, outlier_indices = (
            self.aggregator.aggregate_safe(updates_to_use)
        )

        if outlier_indices:
            logger.warning(
                "Removed %d outlier updates from round %d",
                len(outlier_indices),
                self.current_round,
            )
            n_devices -= len(outlier_indices)

        # Step 3: Apply delta to current model
        current_version, current_weights = self.model_store.get_latest()

        if len(current_weights) == 0:
            # No existing model; delta IS the model
            new_weights = global_delta
        elif len(current_weights) == len(global_delta):
            new_weights = current_weights + global_delta
        else:
            # Dimension mismatch: use delta as new model
            logger.warning(
                "Weight dimensions changed: current=%d, delta=%d",
                len(current_weights),
                len(global_delta),
            )
            new_weights = global_delta

        # Step 4: Validate
        f1, should_accept = self.validator.validate(
            new_weights, self.current_accuracy
        )

        new_version = current_version + 1

        if should_accept:
            # Step 5: Publish
            self.model_store.save_model(
                weights=new_weights,
                version=new_version,
                accuracy=f1,
                n_contributing_devices=n_devices,
            )
            self.current_accuracy = f1
            self.accuracy_history.append(f1)
            self.total_contributing_devices += n_devices

            logger.info(
                "Round %d complete: published model v%d (F1=%.4f, %d devices)",
                self.current_round,
                new_version,
                f1,
                n_devices,
            )
        else:
            logger.warning(
                "Round %d: model v%d REJECTED (F1=%.4f < threshold), "
                "keeping v%d",
                self.current_round,
                new_version,
                f1,
                current_version,
            )

        # Record round history
        self.round_history.append(
            {
                "round_id": self.current_round,
                "n_updates": len(updates_to_use),
                "n_outliers": len(outlier_indices),
                "n_contributing_devices": n_devices,
                "f1": f1,
                "accepted": should_accept,
                "model_version": new_version if should_accept else current_version,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Reset for next round
        self.status = "complete"
        self._reset_round()

        return new_weights

    def get_status(self) -> RoundStatus:
        """Get current round status.

        Returns:
            RoundStatus with current round info.
        """
        current_version, _ = self.model_store.get_latest()
        return RoundStatus(
            round_id=self.current_round,
            status=self.status,
            updates_received=len(self.updates),
            min_updates_required=self.min_devices,
            model_version=current_version,
        )

    def get_global_metrics(self) -> dict:
        """Get aggregated global metrics.

        Returns:
            Dict with total_rounds, current_model_version,
            current_accuracy, total_contributing_devices,
            accuracy_history.
        """
        current_version, _ = self.model_store.get_latest()
        return {
            "total_rounds": len(self.round_history),
            "current_model_version": current_version,
            "current_accuracy": self.current_accuracy,
            "total_contributing_devices": self.total_contributing_devices,
            "accuracy_history": list(self.accuracy_history),
        }

    def _reset_round(self):
        """Reset state for the next round."""
        self.current_round += 1
        self.updates = []
        self._submitted_devices = set()
        self.round_start_time = time.time()
        self.status = "collecting"
        logger.info("Round %d: now collecting updates", self.current_round)
