"""Model stubs for Learning Architecture V2 (optional torch)."""

from playmind.models.policy_v2 import (
    AUX_KEYS,
    DEFAULT_FEATURE_DIM,
    LegacyCheckpointError,
    MODEL_VERSION,
    TORCH_AVAILABLE,
    SkillPolicyNet,
    SkillPolicyV2,
    load_legacy_mlp,
    structured_feature_vector,
    torch_install_instructions,
)
from playmind.models.feature_schema import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FeatureNormalizer,
    structured_feature_vector_v2,
)
from playmind.models.encoders import (
    FusionObservationEncoder,
    StructuredObservationEncoder,
    VisualObservationEncoder,
)
from playmind.models.recurrent_policy import (
    AUX_TYPES,
    DEFAULT_AUX_KEYS,
    RecurrentSkillPolicyNet,
    RecurrentSkillPolicyV2,
    seed_everything,
)

__all__ = [
    "AUX_KEYS",
    "DEFAULT_FEATURE_DIM",
    "DEFAULT_AUX_KEYS",
    "AUX_TYPES",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "FeatureNormalizer",
    "FusionObservationEncoder",
    "LegacyCheckpointError",
    "MODEL_VERSION",
    "RecurrentSkillPolicyNet",
    "RecurrentSkillPolicyV2",
    "StructuredObservationEncoder",
    "TORCH_AVAILABLE",
    "SkillPolicyNet",
    "SkillPolicyV2",
    "VisualObservationEncoder",
    "load_legacy_mlp",
    "seed_everything",
    "structured_feature_vector",
    "structured_feature_vector_v2",
    "torch_install_instructions",
]
