"""Adapter-at-rest protection: AES-256-GCM (an authenticated encryption
construction, per the project brief's requirement — no custom
cryptography, current maintained library: Python `cryptography`, which
wraps OpenSSL's AEAD implementation).

    C_A = AEAD.Enc_K(A_phi, metadata)

`metadata` is bound in as AES-GCM's associated data (AAD): authenticated
but not encrypted, so a policy/version tag can be checked before decrypting
without revealing the adapter weights, and any tampering with the metadata
is detected the same way tampering with the ciphertext is.

RSA is NOT used to encrypt the tensor payload directly (the project brief
explicitly forbids this — RSA has a small message-size limit and no
authentication); if envelope encryption with public-key key-wrapping is
added later, RSA would only wrap the small symmetric key, never the
tensor bytes themselves, and the modern alternative (X25519 + HKDF, or an
RSA-OAEP wrap at minimum) would be used and stated explicitly. Key
distribution/revocation is a SEPARATE mechanism — see
medgate/crypto/authorization.py — this module only encrypts/decrypts
given a key it is handed.
"""
import io
import json

import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_SIZE_BYTES = 32  # AES-256
NONCE_SIZE_BYTES = 12  # standard GCM nonce size


def generate_key() -> bytes:
    return AESGCM.generate_key(bit_length=KEY_SIZE_BYTES * 8)


def encrypt_adapter(state_dict: dict, key: bytes, metadata: dict) -> dict:
    """Serialize an adapter's state_dict with torch.save, then AES-GCM
    encrypt it. `metadata` (e.g. {"model_version": ..., "key_id": ...,
    "policy_hash": ...}) is authenticated (AAD) but stored in the clear
    alongside the ciphertext — the point is integrity/binding, not
    secrecy, of the metadata."""
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    plaintext = buf.getvalue()

    aad = json.dumps(metadata, sort_keys=True).encode()
    nonce = _fresh_nonce()
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, aad)

    return {"nonce": nonce, "ciphertext": ciphertext, "metadata": metadata}


def decrypt_adapter(sealed: dict, key: bytes) -> dict:
    """Raises cryptography.exceptions.InvalidTag if the key is wrong or
    the ciphertext/metadata were tampered with — the caller (
    medgate/crypto/authorization.py) is responsible for checking whether
    the requester is actually authorized BEFORE calling this; this
    function only enforces confidentiality/integrity of the bytes, not
    authorization policy."""
    aad = json.dumps(sealed["metadata"], sort_keys=True).encode()
    plaintext = AESGCM(key).decrypt(sealed["nonce"], sealed["ciphertext"], aad)
    return torch.load(io.BytesIO(plaintext), weights_only=True)


def _fresh_nonce() -> bytes:
    import os
    return os.urandom(NONCE_SIZE_BYTES)
