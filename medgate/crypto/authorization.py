"""Minimal centralized IAM: an in-memory credential registry backing the
adapter-encryption layer (medgate/crypto/adapter_encryption.py). This is
the "conventional centralized IAM baseline" the project brief asks to
compare against an optional permissioned-ledger authorization (Phase 8,
not yet implemented — see docs/execution_plan.md).

Deliberately NOT a production auth system: no persistence, no network, no
password/MFA handling, no CP-ABE policy evaluation — just enough to give
Phase 5 a real revocation mechanism to test the project brief's
pre-registered criterion "expired/revoked credential acceptance rate of
zero" against, and to give Phase 8 a real (if minimal) baseline to compare
a ledger-based scheme to on latency/audit-consistency terms.
"""
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Credential:
    subject_id: str
    key: bytes
    issued_at: float
    revoked: bool = False
    revoked_at: float | None = None
    expires_at: float | None = None  # None = no expiry


class CentralizedIAM:
    """key_id -> Credential. `audit_log` records every issue/revoke/access
    decision with a timestamp — the "append-only audit log" half of the
    project brief's "centralized IAM plus append-only audit log" baseline."""

    def __init__(self):
        self._credentials: dict[str, Credential] = {}
        self.audit_log: list[dict] = []

    def issue(self, subject_id: str, key: bytes, ttl_seconds: float | None = None) -> str:
        start = time.perf_counter()
        key_id = str(uuid.uuid4())
        now = time.time()
        expires_at = now + ttl_seconds if ttl_seconds is not None else None
        self._credentials[key_id] = Credential(subject_id=subject_id, key=key, issued_at=now, expires_at=expires_at)
        latency = time.perf_counter() - start
        self.audit_log.append({"event": "issue", "key_id": key_id, "subject_id": subject_id, "latency_s": latency})
        return key_id

    def revoke(self, key_id: str) -> float:
        """Returns revocation latency in seconds (a Phase 8 metric)."""
        start = time.perf_counter()
        cred = self._credentials.get(key_id)
        if cred is None:
            raise KeyError(f"unknown key_id {key_id}")
        cred.revoked = True
        cred.revoked_at = time.time()
        latency = time.perf_counter() - start
        self.audit_log.append({"event": "revoke", "key_id": key_id, "latency_s": latency})
        return latency

    def rotate(self, old_key_id: str, new_key: bytes, ttl_seconds: float | None = None) -> str:
        """Key rotation = revoke the old credential + issue a fresh one
        for the same subject (P1-11 review requirement: 'explicit
        key-rotation ... behavior'). Does NOT re-encrypt any adapter
        already sealed under the old key — that is a separate,
        caller-driven step (decrypt with the old key while it's still
        valid, i.e. BEFORE calling rotate, then re-encrypt with the
        returned new key_id's key) — this method only handles the
        credential lifecycle, matching medgate/crypto/adapter_encryption.py's
        own separation of concerns (encryption vs. key distribution)."""
        cred = self._credentials.get(old_key_id)
        if cred is None:
            raise KeyError(f"unknown key_id {old_key_id}")
        subject_id = cred.subject_id
        self.revoke(old_key_id)
        return self.issue(subject_id, new_key, ttl_seconds=ttl_seconds)

    def authorize(self, key_id: str, now: float | None = None) -> bytes:
        """Returns the key IF the credential exists, is not revoked, and
        (if issued with a TTL) has not expired; raises PermissionError
        otherwise. `now` is injectable for tests that simulate the clock
        advancing past expiry without a real sleep. The pre-registered
        success criterion (docs/research_scope.md): revoked/expired
        credential acceptance rate must be exactly zero — every caller of
        this class in tests/scripts checks that a revoked OR expired
        key_id always raises here, never silently returns a key. This is
        also the mandatory 'replay of expired authorization tokens'
        integrity check: presenting an expired key_id again (replaying it)
        must be denied every time, not just the first time after expiry."""
        start = time.perf_counter()
        now = time.time() if now is None else now
        cred = self._credentials.get(key_id)
        expired = cred is not None and cred.expires_at is not None and now >= cred.expires_at
        allowed = cred is not None and not cred.revoked and not expired
        latency = time.perf_counter() - start
        self.audit_log.append({
            "event": "authorize", "key_id": key_id, "allowed": allowed,
            "expired": expired, "latency_s": latency,
        })
        if not allowed:
            reason = "expired" if expired else "unknown or revoked"
            raise PermissionError(f"key_id {key_id} is {reason}")
        return cred.key
