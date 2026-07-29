"""PlayMind Offline Studio.

This package processes files only. It intentionally has no live capture,
process-access, physical-input logging, or generated-input surface.
"""

from playmind.studio.profiles import (
    PROFILE_RETAIL_WOW_OFFLINE_ONLY,
    ProtectedProfile,
)
from playmind.studio.provenance import ProvenanceRecord, is_training_eligible
from playmind.studio.safety import (
    StudioSafetyError,
    assert_studio_safe,
    studio_may_not_send_input,
)

__all__ = [
    "PROFILE_RETAIL_WOW_OFFLINE_ONLY",
    "ProtectedProfile",
    "ProvenanceRecord",
    "StudioSafetyError",
    "assert_studio_safe",
    "is_training_eligible",
    "studio_may_not_send_input",
]
