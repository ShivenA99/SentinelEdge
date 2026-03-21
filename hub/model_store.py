"""Model store with versioning, Ed25519 signing, and delta compression.

Manages the lifecycle of global model weights:
- Versioned storage of model weights and metadata
- Ed25519 signing so edge devices can verify authenticity
- Compressed delta patches for bandwidth-efficient distribution
"""

import base64
import gzip
import json
import logging
import os
from datetime import datetime, timezone

import nacl.encoding
import nacl.signing
import numpy as np

logger = logging.getLogger(__name__)


class ModelStore:
    """Versioned model store with Ed25519 signing and delta compression."""

    def __init__(
        self,
        store_dir: str = "model_store",
        private_key_path: str | None = None,
    ):
        """Initialize the model store.

        Args:
            store_dir: Directory to store model versions and metadata.
            private_key_path: Path to Ed25519 private key file.
                If None, defaults to <store_dir>/signing_key.bin.
        """
        self.store_dir = store_dir
        self.private_key_path = private_key_path or os.path.join(
            store_dir, "signing_key.bin"
        )

        # In-memory state
        self._models: dict[int, np.ndarray] = {}       # version -> weights
        self._metadata: dict[int, dict] = {}            # version -> metadata
        self._latest_version: int = 0

        self._signing_key: nacl.signing.SigningKey | None = None
        self._verify_key: nacl.signing.VerifyKey | None = None

        # Ensure store directory exists
        os.makedirs(self.store_dir, exist_ok=True)

        # Initialize signing keys
        self.initialize_keys()

    def initialize_keys(self):
        """Load existing Ed25519 keypair or generate a new one.

        Keys are persisted to disk so they survive server restarts.
        """
        if os.path.exists(self.private_key_path):
            try:
                with open(self.private_key_path, "rb") as f:
                    seed = f.read()
                self._signing_key = nacl.signing.SigningKey(seed)
                self._verify_key = self._signing_key.verify_key
                logger.info(
                    "Loaded Ed25519 signing key from %s", self.private_key_path
                )
                return
            except Exception as e:
                logger.warning(
                    "Failed to load signing key from %s: %s, generating new",
                    self.private_key_path,
                    e,
                )

        # Generate new keypair
        self._signing_key = nacl.signing.SigningKey.generate()
        self._verify_key = self._signing_key.verify_key

        # Persist the seed (32 bytes)
        try:
            with open(self.private_key_path, "wb") as f:
                f.write(bytes(self._signing_key))
            logger.info("Generated and saved new Ed25519 key to %s", self.private_key_path)
        except Exception as e:
            logger.warning(
                "Could not persist signing key to %s: %s (operating in-memory only)",
                self.private_key_path,
                e,
            )

    def save_model(
        self,
        weights: np.ndarray,
        version: int,
        accuracy: float,
        n_contributing_devices: int = 0,
    ):
        """Save model weights, sign them, and store metadata.

        Args:
            weights: Model weight vector.
            version: Version number to assign.
            accuracy: Validation accuracy / F1 for this version.
            n_contributing_devices: How many edge devices contributed.
        """
        self._models[version] = np.array(weights, dtype=np.float64).copy()

        model_bytes = weights.astype(np.float64).tobytes()
        signature = self.sign_model(model_bytes)

        metadata = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "accuracy": accuracy,
            "n_contributing_devices": n_contributing_devices,
            "signature": signature,
            "n_params": len(weights),
        }
        self._metadata[version] = metadata

        if version > self._latest_version:
            self._latest_version = version

        # Persist to disk as well
        try:
            version_dir = os.path.join(self.store_dir, f"v{version}")
            os.makedirs(version_dir, exist_ok=True)

            np.save(os.path.join(version_dir, "weights.npy"), weights)

            with open(os.path.join(version_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                "Saved model v%d (accuracy=%.4f, %d params) to %s",
                version,
                accuracy,
                len(weights),
                version_dir,
            )
        except Exception as e:
            logger.warning("Could not persist model v%d to disk: %s", version, e)

    def get_latest(self) -> tuple[int, np.ndarray]:
        """Get latest model version and weights.

        Returns:
            Tuple of (version, weights). Returns (0, empty_array) if no model exists.
        """
        if self._latest_version == 0 or self._latest_version not in self._models:
            return 0, np.array([], dtype=np.float64)

        return self._latest_version, self._models[self._latest_version].copy()

    def get_model(self, version: int) -> np.ndarray | None:
        """Get weights for a specific version.

        Args:
            version: Model version to retrieve.

        Returns:
            Weight array, or None if version not found.
        """
        if version in self._models:
            return self._models[version].copy()
        return None

    def get_metadata(self, version: int) -> dict | None:
        """Get metadata for a specific version.

        Args:
            version: Model version.

        Returns:
            Metadata dict, or None if version not found.
        """
        return self._metadata.get(version)

    def create_delta_patch(self, old_version: int, new_version: int) -> bytes:
        """Create compressed delta patch between two model versions.

        The patch is: gzip(new_weights - old_weights as float64 bytes).
        This typically compresses well since deltas are small.

        If old_version is 0 or not found, the full new weights are used
        as the delta (i.e., the patch is the complete model).

        Args:
            old_version: Base version for the delta.
            new_version: Target version.

        Returns:
            Gzipped delta bytes.

        Raises:
            ValueError: If new_version not found in store.
        """
        if new_version not in self._models:
            raise ValueError(f"Model version {new_version} not found")

        new_weights = self._models[new_version]

        if old_version > 0 and old_version in self._models:
            old_weights = self._models[old_version]
            # Handle dimension changes gracefully
            if len(old_weights) == len(new_weights):
                delta = new_weights - old_weights
            else:
                logger.warning(
                    "Dimension mismatch between v%d (%d) and v%d (%d), "
                    "sending full weights",
                    old_version,
                    len(old_weights),
                    new_version,
                    len(new_weights),
                )
                delta = new_weights
        else:
            delta = new_weights

        delta_bytes = delta.astype(np.float64).tobytes()
        compressed = gzip.compress(delta_bytes, compresslevel=9)

        logger.info(
            "Delta patch v%d->v%d: %d bytes raw, %d bytes compressed (%.1f%% reduction)",
            old_version,
            new_version,
            len(delta_bytes),
            len(compressed),
            100.0 * (1 - len(compressed) / max(len(delta_bytes), 1)),
        )
        return compressed

    def create_delta_patch_b64(self, old_version: int, new_version: int) -> str:
        """Create a base64-encoded compressed delta patch.

        Convenience wrapper around create_delta_patch for API responses.

        Args:
            old_version: Base version.
            new_version: Target version.

        Returns:
            Base64 string of the gzipped delta.
        """
        compressed = self.create_delta_patch(old_version, new_version)
        return base64.b64encode(compressed).decode("ascii")

    def sign_model(self, model_bytes: bytes) -> str:
        """Sign model bytes with Ed25519 private key.

        Args:
            model_bytes: Raw bytes to sign.

        Returns:
            Hex-encoded signature string.
        """
        if self._signing_key is None:
            raise RuntimeError("Signing key not initialized")

        signed = self._signing_key.sign(model_bytes)
        # signed.signature is the 64-byte signature
        return signed.signature.hex()

    def get_public_key(self) -> str:
        """Return public key hex for edge devices to verify signatures.

        Returns:
            Hex-encoded Ed25519 public key (32 bytes = 64 hex chars).
        """
        if self._verify_key is None:
            raise RuntimeError("Signing key not initialized")

        return self._verify_key.encode(encoder=nacl.encoding.HexEncoder).decode("ascii")

    def get_signature_for_version(self, version: int) -> str:
        """Get the stored signature for a model version.

        Args:
            version: Model version.

        Returns:
            Hex signature string.

        Raises:
            ValueError: If version not found.
        """
        meta = self._metadata.get(version)
        if meta is None:
            raise ValueError(f"No metadata for version {version}")
        return meta["signature"]

    def list_versions(self) -> list[int]:
        """List all stored model versions in ascending order."""
        return sorted(self._models.keys())
