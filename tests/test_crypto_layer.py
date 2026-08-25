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


# --------------------------------------------------------------- P1-11 additions

def test_nonces_are_fresh_and_unique_per_encryption():
    """Same key, same plaintext, same metadata, encrypted twice: the
    nonce and ciphertext must differ both times (AES-GCM's security
    depends on never reusing a nonce under the same key)."""
    key = generate_key()
    state = _dummy_state_dict()
    sealed_1 = encrypt_adapter(state, key, metadata={"model_version": "v1"})
    sealed_2 = encrypt_adapter(state, key, metadata={"model_version": "v1"})
    assert sealed_1["nonce"] != sealed_2["nonce"]
    assert sealed_1["ciphertext"] != sealed_2["ciphertext"]
    # both still decrypt correctly under the same key -- freshness doesn't break correctness
    assert torch.equal(decrypt_adapter(sealed_1, key)["down.weight"], state["down.weight"])
    assert torch.equal(decrypt_adapter(sealed_2, key)["down.weight"], state["down.weight"])


def test_generated_keys_are_unique_and_correct_length():
    keys = [generate_key() for _ in range(20)]
    assert len(set(keys)) == 20, "generate_key() produced a duplicate -- RNG not behaving as expected"
    assert all(len(k) == 32 for k in keys)  # AES-256


def test_ciphertext_bit_flip_is_detected():
    """Distinct from the metadata-tamper test: this corrupts the
    CIPHERTEXT itself, not the AAD."""
    key = generate_key()
    sealed = encrypt_adapter(_dummy_state_dict(), key, metadata={"model_version": "v1"})
    corrupted = bytearray(sealed["ciphertext"])
    corrupted[0] ^= 0xFF
    sealed["ciphertext"] = bytes(corrupted)
    with pytest.raises(InvalidTag):
        decrypt_adapter(sealed, key)


def test_truncated_ciphertext_is_rejected_not_silently_accepted():
    key = generate_key()
    sealed = encrypt_adapter(_dummy_state_dict(), key, metadata={"model_version": "v1"})
    sealed["ciphertext"] = sealed["ciphertext"][:-4]
    with pytest.raises((InvalidTag, ValueError)):
        decrypt_adapter(sealed, key)


def test_every_metadata_field_is_bound_by_aad_not_just_one():
    """model_id, adapter_version, policy_version, ... (P1-11: 'model ID,
    adapter version, policy version, and tensor metadata bound through
    AAD') -- tampering with ANY one of several metadata fields must break
    decryption, not just the one field an earlier test happened to check."""
    key = generate_key()
    metadata = {"model_id": "mg-1", "adapter_version": "3", "policy_version": "p7", "key_id": "k1"}
    sealed = encrypt_adapter(_dummy_state_dict(), key, metadata=metadata)
    for field in metadata:
        tampered = dict(sealed)
        tampered["metadata"] = {**metadata, field: metadata[field] + "-tampered"}
        with pytest.raises(InvalidTag):
            decrypt_adapter(tampered, key)


def test_sealed_output_never_contains_the_raw_key():
    """No key or plaintext adapter printed/returned outside what's needed
    (P1-11) -- encrypt_adapter's return value must not leak the key
    itself (a caller who logs `sealed` should never accidentally log
    key material)."""
    key = generate_key()
    sealed = encrypt_adapter(_dummy_state_dict(), key, metadata={"model_version": "v1"})
    serialized = repr(sealed) + repr(sealed.get("metadata"))
    assert key.hex() not in serialized
    assert key not in sealed.values()


def test_key_rotation_workflow_revokes_old_and_reencrypts_under_new_key():
    """Key rotation = revoke old credential + issue new one
    (CentralizedIAM.rotate) + caller re-encrypts with the new key. The old
    ciphertext must NOT decrypt under the new key (rotation is not
    automatic re-encryption); the old key_id must be rejected afterward."""
    iam = CentralizedIAM()
    old_key = generate_key()
    old_key_id = iam.issue(subject_id="hospital_c", key=old_key)
    state = _dummy_state_dict()
    sealed_old = encrypt_adapter(state, old_key, metadata={"model_version": "v1", "key_id": old_key_id})

    new_key = generate_key()
    new_key_id = iam.rotate(old_key_id, new_key)

    with pytest.raises(PermissionError):
        iam.authorize(old_key_id)  # old credential is now revoked
    assert iam.authorize(new_key_id) == new_key

    with pytest.raises(InvalidTag):
        decrypt_adapter(sealed_old, new_key)  # rotation does not retroactively re-encrypt

    sealed_new = encrypt_adapter(state, new_key, metadata={"model_version": "v1", "key_id": new_key_id})
    recovered = decrypt_adapter(sealed_new, new_key)
    assert torch.equal(recovered["down.weight"], state["down.weight"])


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
    test_nonces_are_fresh_and_unique_per_encryption()
    test_generated_keys_are_unique_and_correct_length()
    test_ciphertext_bit_flip_is_detected()
    test_truncated_ciphertext_is_rejected_not_silently_accepted()
    test_every_metadata_field_is_bound_by_aad_not_just_one()
    test_sealed_output_never_contains_the_raw_key()
    test_key_rotation_workflow_revokes_old_and_reencrypts_under_new_key()
    test_audit_log_records_every_event()
    print("OK")
