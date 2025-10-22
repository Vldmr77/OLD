"""Simple symmetric encryption helper based on Fernet keys."""
from __future__ import annotations

import base64
import binascii
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - prefer cryptography if available
    from cryptography.fernet import Fernet, InvalidToken
except Exception:  # pragma: no cover - fallback cipher for offline environments

    class InvalidToken(Exception):
        """Raised when the fallback cipher cannot decrypt a token."""

    class Fernet:  # type: ignore
        """Lightweight XOR-based fallback compatible with Fernet API."""

        def __init__(self, key: bytes) -> None:
            try:
                raw = base64.urlsafe_b64decode(key)
            except (ValueError, binascii.Error) as exc:  # pragma: no cover - invalid input branch
                raise ValueError("Invalid key material") from exc
            if len(raw) != 32:
                raise ValueError("Unexpected key length")
            self._key = hashlib.sha256(raw).digest()

        @staticmethod
        def generate_key() -> bytes:
            return base64.urlsafe_b64encode(os.urandom(32))

        def encrypt(self, data: bytes) -> bytes:
            nonce = os.urandom(16)
            stream = self._keystream(nonce, len(data))
            ciphertext = bytes(b ^ s for b, s in zip(data, stream))
            return base64.urlsafe_b64encode(nonce + ciphertext)

        def decrypt(self, token: bytes) -> bytes:
            try:
                raw = base64.urlsafe_b64decode(token)
            except (ValueError, binascii.Error) as exc:  # pragma: no cover - invalid token branch
                raise InvalidToken("Malformed token") from exc
            if len(raw) < 16:
                raise InvalidToken("Token too short")
            nonce, ciphertext = raw[:16], raw[16:]
            stream = self._keystream(nonce, len(ciphertext))
            return bytes(c ^ s for c, s in zip(ciphertext, stream))

        def _keystream(self, nonce: bytes, length: int) -> bytes:
            stream = bytearray()
            counter = 0
            while len(stream) < length:
                counter_bytes = counter.to_bytes(4, "big")
                digest = hashlib.sha256(self._key + nonce + counter_bytes).digest()
                stream.extend(digest)
                counter += 1
            return bytes(stream[:length])


@dataclass
class KeyManager:
    """Manage encryption keys and decrypt secrets stored in configs."""

    key: bytes

    @classmethod
    def generate(cls) -> "KeyManager":
        """Generate a new random key."""

        return cls(Fernet.generate_key())

    @classmethod
    def from_key_string(cls, value: str) -> "KeyManager":
        """Create a key manager from a string representation.

        The string may already be a url-safe base64 encoded key or an
        ascii-armoured token produced by :meth:`KeyManager.serialise`.
        """

        value = value.strip()
        try:
            decoded = base64.urlsafe_b64decode(value)
        except (ValueError, binascii.Error) as exc:  # pragma: no cover - invalid input branch
            raise ValueError("Invalid key material") from exc
        if len(decoded) != 32:
            # Fernet expects 32 raw bytes before base64 encoding
            raise ValueError("Unexpected key length")
        return cls(value.encode("utf-8"))

    @classmethod
    def from_file(cls, path: Path) -> "KeyManager":
        """Load a key from the given file path."""

        return cls.from_key_string(path.read_text(encoding="utf-8"))

    def serialise(self) -> str:
        """Return the string representation of the key."""

        return self.key.decode("utf-8")

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext string using Fernet."""

        token = Fernet(self.key).encrypt(plaintext.encode("utf-8"))
        return token.decode("utf-8")

    def decrypt(self, token: str, *, default: Optional[str] = None) -> str:
        """Decrypt the provided token returning unicode text.

        Parameters
        ----------
        token:
            The base64 url-safe cipher text.
        default:
            Optional default to return when the token cannot be decrypted.
        """

        fernet = Fernet(self.key)
        try:
            decrypted = fernet.decrypt(token.encode("utf-8"))
        except InvalidToken:
            if default is not None:
                return default
            raise
        return decrypted.decode("utf-8")

    def maybe_decrypt(self, value: Optional[str]) -> Optional[str]:
        """Decrypt tokens prefixed with ``enc:`` leaving others untouched."""

        if value is None:
            return None
        if value.startswith("enc:"):
            return self.decrypt(value[4:])
        return value


__all__ = ["KeyManager"]
