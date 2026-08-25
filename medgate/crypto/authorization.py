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


class CentralizedIAM:
    """key_id -> Credential. `audit_log` records every issue/revoke/access
    decision with a timestamp — the "append-only audit log" half of the
    project brief's "centralized IAM plus append-only audit log" baseline."""

    def __init__(self):
        self._credentials: dict[str, Credential] = {}
        self.audit_log: list[dict] = []

    def issue(self, subject_id: str, key: bytes) -> str:
        start = time.perf_counter()
        key_id = str(uuid.uuid4())
        self._credentials[key_id] = Credential(subject_id=subject_id, key=key, issued_at=time.time())
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

    def authorize(self, key_id: str) -> bytes:
        """Returns the key IF the credential exists and is not revoked;
        raises PermissionError otherwise. The pre-registered success
        criterion (docs/research_scope.md): revoked-credential acceptance
        rate must be exactly zero — every caller of this class in
        tests/scripts checks that a revoked key_id always raises here,
        never silently returns a key."""
        start = time.perf_counter()
        cred = self._credentials.get(key_id)
        allowed = cred is not None and not cred.revoked
        latency = time.perf_counter() - start
        self.audit_log.append({"event": "authorize", "key_id": key_id, "allowed": allowed, "latency_s": latency})
        if not allowed:
            raise PermissionError(f"key_id {key_id} is unknown or revoked")
        return cred.key
