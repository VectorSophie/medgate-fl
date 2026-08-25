"""Checks for the crypto/authorization layer (docs/research_scope.md
"Cryptographic and authorization layer"): AES-GCM roundtrip + tamper
detection, and the pre-registered zero-acceptance-of-revoked-credentials
criterion.

Run: PYTHONPATH=. pytest tests/test_crypto_layer.py -v
"""
import pytest
import torch
from cryptography.exceptions import InvalidTag

from medgate.crypto.adapter_encryption import decrypt_adapter, encrypt_adapter, generate_key
from medgate.crypto.authorization import CentralizedIAM


def _dummy_state_dict():
    return {"down.weight": torch.randn(4, 8), "up.weight": torch.zeros(8, 4)}


def test_encrypt_decrypt_roundtrip_is_exact():
    key = generate_key()
    state = _dummy_state_dict()
    sealed = encrypt_adapter(state, key, metadata={"model_version": "v1", "key_id": "k1"})
    recovered = decrypt_adapter(sealed, key)
    for k in state:
        assert torch.equal(state[k], recovered[k])


def test_wrong_key_fails_to_decrypt():
    state = _dummy_state_dict()
    sealed = encrypt_adapter(state, generate_key(), metadata={"model_version": "v1"})
    with pytest.raises(InvalidTag):
        decrypt_adapter(sealed, generate_key())


def test_tampered_metadata_is_detected():
    key = generate_key()
    state = _dummy_state_dict()
    sealed = encrypt_adapter(state, key, metadata={"model_version": "v1"})
    sealed["metadata"]["model_version"] = "v2"  # tamper with the authenticated (but unencrypted) metadata
    with pytest.raises(InvalidTag):
        decrypt_adapter(sealed, key)


def test_revoked_credential_is_never_accepted():
    """Directly tests the pre-registered criterion in docs/research_scope.md:
    'expired/revoked credential acceptance rate of zero'."""
    iam = CentralizedIAM()
    key = generate_key()
    key_id = iam.issue(subject_id="hospital_a", key=key)
    assert iam.authorize(key_id) == key  # valid before revocation

    iam.revoke(key_id)
    attempts = 20
    denied = 0
    for _ in range(attempts):
        try:
            iam.authorize(key_id)
        except PermissionError:
            denied += 1
    assert denied == attempts, f"revoked credential was accepted {attempts - denied}/{attempts} times"


def test_unknown_key_id_is_denied():
    iam = CentralizedIAM()
    with pytest.raises(PermissionError):
        iam.authorize("no-such-key")


def test_audit_log_records_every_event():
    iam = CentralizedIAM()
    key_id = iam.issue(subject_id="hospital_b", key=generate_key())
    iam.authorize(key_id)
    iam.revoke(key_id)
    events = [e["event"] for e in iam.audit_log]
    assert events == ["issue", "authorize", "revoke"]


if __name__ == "__main__":
    test_encrypt_decrypt_roundtrip_is_exact()
    test_wrong_key_fails_to_decrypt()
    test_tampered_metadata_is_detected()
    test_revoked_credential_is_never_accepted()
    test_unknown_key_id_is_denied()
    test_audit_log_records_every_event()
    print("OK")
