"""Source provenance and conservative training eligibility."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


SOURCE_USER_OWNED_RECORDING = "user_owned_recording"
SOURCE_FRIEND_PROVIDED = "friend_provided"
SOURCE_CONSENTING_CREATOR = "consenting_creator"
SOURCE_LICENSED_DATASET = "licensed_dataset"
SOURCE_SYNTHETIC = "synthetic"
SOURCE_UNKNOWN = "unknown"
SOURCE_TYPES = frozenset(
    {
        SOURCE_USER_OWNED_RECORDING,
        SOURCE_FRIEND_PROVIDED,
        SOURCE_CONSENTING_CREATOR,
        SOURCE_LICENSED_DATASET,
        SOURCE_SYNTHETIC,
        SOURCE_UNKNOWN,
    }
)


@dataclass
class ProvenanceRecord:
    source_type: str
    source_id: str = ""
    rights_confirmed: bool = False
    permission_confirmed: bool = False
    consent_confirmed: bool = False
    license_confirmed: bool = False
    training_use_allowed: bool = False
    private_use_only: bool = False
    attribution: str = ""
    license_name: str = ""
    source_uri: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_type = str(self.source_type).strip().lower()
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProvenanceRecord:
        return cls(**dict(value))

    @classmethod
    def from_json(cls, value: str) -> ProvenanceRecord:
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise ValueError("provenance JSON must contain an object")
        return cls.from_dict(decoded)


def coerce_provenance(value: ProvenanceRecord | Mapping[str, Any]) -> ProvenanceRecord:
    return value if isinstance(value, ProvenanceRecord) else ProvenanceRecord.from_dict(value)


def is_training_eligible(
    record: ProvenanceRecord | Mapping[str, Any],
    *,
    allow_unverified_private: bool = False,
) -> bool:
    """Return eligibility only when source rights/consent are explicit.

    ``allow_unverified_private`` is intentionally narrow: it permits an
    unverified user-owned recording marked private, never unknown/public data.
    """

    item = coerce_provenance(record)
    rights = item.rights_confirmed or item.permission_confirmed
    training_rights = rights or item.training_use_allowed
    if item.source_type == SOURCE_SYNTHETIC:
        return True
    if item.source_type == SOURCE_UNKNOWN:
        return False
    if item.source_type == SOURCE_LICENSED_DATASET:
        return training_rights and item.license_confirmed
    if item.source_type in {SOURCE_FRIEND_PROVIDED, SOURCE_CONSENTING_CREATOR}:
        return training_rights and item.consent_confirmed
    if item.source_type == SOURCE_USER_OWNED_RECORDING:
        return training_rights or (
            allow_unverified_private and item.private_use_only
        )
    return False


__all__ = [
    "ProvenanceRecord",
    "SOURCE_CONSENTING_CREATOR",
    "SOURCE_FRIEND_PROVIDED",
    "SOURCE_LICENSED_DATASET",
    "SOURCE_SYNTHETIC",
    "SOURCE_TYPES",
    "SOURCE_UNKNOWN",
    "SOURCE_USER_OWNED_RECORDING",
    "coerce_provenance",
    "is_training_eligible",
]
