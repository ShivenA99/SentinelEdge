"""Federated Averaging aggregator with Byzantine fault detection."""

import logging

import numpy as np

from .schemas import FederatedUpdate

logger = logging.getLogger(__name__)


class FedAvgAggregator:
    """Federated Averaging: weighted mean of DP-noised gradient deltas.

    Implements the FedAvg algorithm from McMahan et al. (2017) with
    additional outlier detection for Byzantine fault tolerance.
    """

    def aggregate(self, updates: list[FederatedUpdate]) -> np.ndarray:
        """Compute weighted average of gradient deltas.

        Weight = n_samples for each device.
        G_global = sum(n_i * G_i) / sum(n_i)

        Args:
            updates: List of federated updates from edge devices.

        Returns:
            Weighted average gradient delta as a numpy array.

        Raises:
            ValueError: If updates list is empty or gradient dimensions mismatch.
        """
        if not updates:
            raise ValueError("Cannot aggregate empty updates list")

        dim = len(updates[0].gradient_delta)
        for i, u in enumerate(updates):
            if len(u.gradient_delta) != dim:
                raise ValueError(
                    f"Gradient dimension mismatch: update 0 has {dim}, "
                    f"update {i} has {len(u.gradient_delta)}"
                )

        total_samples = sum(u.n_samples for u in updates)
        if total_samples == 0:
            raise ValueError("Total samples across all updates is zero")

        weighted_sum = np.zeros(dim, dtype=np.float64)
        for update in updates:
            delta = np.array(update.gradient_delta, dtype=np.float64)
            weighted_sum += update.n_samples * delta

        global_delta = weighted_sum / total_samples

        logger.info(
            "Aggregated %d updates (%d total samples), "
            "delta norm=%.6f",
            len(updates),
            total_samples,
            float(np.linalg.norm(global_delta)),
        )
        return global_delta

    def aggregate_feature_importances(
        self, updates: list[FederatedUpdate]
    ) -> np.ndarray:
        """Weighted averaging for feature importances.

        Same weighting scheme as gradient aggregation: weight = n_samples.

        Args:
            updates: List of federated updates from edge devices.

        Returns:
            Weighted average feature importances as a numpy array.

        Raises:
            ValueError: If updates list is empty or dimensions mismatch.
        """
        if not updates:
            raise ValueError("Cannot aggregate empty updates list")

        dim = len(updates[0].feature_importances)
        for i, u in enumerate(updates):
            if len(u.feature_importances) != dim:
                raise ValueError(
                    f"Feature importance dimension mismatch: update 0 has {dim}, "
                    f"update {i} has {len(u.feature_importances)}"
                )

        total_samples = sum(u.n_samples for u in updates)
        if total_samples == 0:
            raise ValueError("Total samples across all updates is zero")

        weighted_sum = np.zeros(dim, dtype=np.float64)
        for update in updates:
            importances = np.array(update.feature_importances, dtype=np.float64)
            weighted_sum += update.n_samples * importances

        global_importances = weighted_sum / total_samples
        return global_importances

    def detect_outliers(
        self, updates: list[FederatedUpdate], threshold: float = 3.0
    ) -> list[int]:
        """Detect potential Byzantine/poisoning updates.

        Flag updates where the gradient L2 norm is more than `threshold`
        standard deviations from the mean norm across all updates.

        Args:
            updates: List of federated updates from edge devices.
            threshold: Number of standard deviations for outlier detection.

        Returns:
            List of indices of suspicious updates.
        """
        if len(updates) < 2:
            return []

        norms = np.array(
            [np.linalg.norm(u.gradient_delta) for u in updates], dtype=np.float64
        )

        mean_norm = float(np.mean(norms))
        std_norm = float(np.std(norms))

        if std_norm < 1e-10:
            # All norms are essentially the same -- no outliers
            return []

        outlier_indices = []
        for i, norm in enumerate(norms):
            z_score = abs(norm - mean_norm) / std_norm
            if z_score > threshold:
                logger.warning(
                    "Outlier detected: update %d from device %s, "
                    "norm=%.4f, z_score=%.2f (threshold=%.2f)",
                    i,
                    updates[i].device_id,
                    norm,
                    z_score,
                    threshold,
                )
                outlier_indices.append(i)

        return outlier_indices

    def aggregate_safe(
        self, updates: list[FederatedUpdate], outlier_threshold: float = 3.0
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Aggregate with automatic outlier removal.

        Detects outliers, removes them, then aggregates the remaining
        updates. If too few updates remain after outlier removal, uses
        all updates anyway with a warning.

        Args:
            updates: List of federated updates.
            outlier_threshold: Z-score threshold for outlier detection.

        Returns:
            Tuple of (global_delta, feature_importances, outlier_indices).
        """
        outlier_indices = self.detect_outliers(updates, outlier_threshold)

        if outlier_indices:
            clean_updates = [
                u for i, u in enumerate(updates) if i not in outlier_indices
            ]
            if len(clean_updates) < 2:
                logger.warning(
                    "Too many outliers (%d/%d) -- using all updates",
                    len(outlier_indices),
                    len(updates),
                )
                clean_updates = updates
                outlier_indices = []
            else:
                logger.info(
                    "Removed %d outlier updates, aggregating %d remaining",
                    len(outlier_indices),
                    len(clean_updates),
                )
        else:
            clean_updates = updates

        global_delta = self.aggregate(clean_updates)
        feature_importances = self.aggregate_feature_importances(clean_updates)

        return global_delta, feature_importances, outlier_indices
