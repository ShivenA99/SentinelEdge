"""Gradient delta computation for federated learning.

Computes the difference between local model weights (trained on the
edge device) and the current global model weights, and applies
aggregated deltas back to the global model.  These operations form
the core of the FedAvg / FedSGD communication protocol.
"""

from __future__ import annotations

import numpy as np


class GradientDeltaComputer:
    """Stateless helper for computing and applying weight deltas.

    In the federated learning round-trip:

    1. The edge device downloads global weights.
    2. Local fine-tuning produces updated local weights.
    3. ``compute_delta`` calculates the difference.
    4. The delta is (optionally) clipped and noised by ``DPNoiseInjector``.
    5. The noisy delta is sent to the central hub.
    6. The hub aggregates deltas from multiple devices.
    7. ``apply_delta`` updates the global model.
    """

    def compute_delta(
        self,
        local_weights: np.ndarray,
        global_weights: np.ndarray,
    ) -> np.ndarray:
        """Compute the weight delta (local minus global).

        Parameters
        ----------
        local_weights : np.ndarray
            Weights after local training on the edge device.
        global_weights : np.ndarray
            Current global model weights.

        Returns
        -------
        np.ndarray
            ``local_weights - global_weights``, same shape and dtype.
        """
        return local_weights - global_weights

    def apply_delta(
        self,
        global_weights: np.ndarray,
        aggregated_delta: np.ndarray,
    ) -> np.ndarray:
        """Apply an aggregated delta to the global weights.

        Parameters
        ----------
        global_weights : np.ndarray
            Current global model weights.
        aggregated_delta : np.ndarray
            Mean (or weighted average) of per-device deltas received
            by the hub.

        Returns
        -------
        np.ndarray
            Updated global weights.
        """
        return global_weights + aggregated_delta
