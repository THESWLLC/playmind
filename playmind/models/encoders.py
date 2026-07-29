"""Modular observation encoders for structured (and future visual) inputs.

RecurrentSkillPolicy consumes per-timestep embeddings from an ObservationEncoder.
Visual learning is NOT implemented — VisualObservationEncoder is a placeholder.
"""

from __future__ import annotations

from typing import Any, Sequence

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class ObservationEncoder(nn.Module):  # type: ignore[misc]
        """Maps per-timestep features → embedding (B, T, E) or (B, E)."""

        def __init__(self, input_dim: int, embed_dim: int) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.embed_dim = int(embed_dim)

        def forward(self, batch: Any) -> Any:
            raise NotImplementedError

    class StructuredObservationEncoder(ObservationEncoder):
        """Linear + LayerNorm + ReLU encoder over structured feature vectors."""

        def __init__(
            self,
            input_dim: int,
            embed_dim: int = 96,
            dropout: float = 0.0,
        ) -> None:
            super().__init__(input_dim, embed_dim)
            self.net = nn.Sequential(
                nn.Linear(self.input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )

        def forward(self, batch: Any) -> Any:
            # batch: (B, T, F) or (B, F)
            return self.net(batch)

    class VisualObservationEncoder(ObservationEncoder):
        """Placeholder — does not consume frames yet.

        Raising is intentional until synchronized frame datasets exist.
        """

        def __init__(self, input_dim: int = 0, embed_dim: int = 96) -> None:
            super().__init__(input_dim=max(1, input_dim), embed_dim=embed_dim)
            self._note = "visual encoder not implemented"

        def forward(self, batch: Any) -> Any:
            raise NotImplementedError(
                "VisualObservationEncoder is a placeholder. "
                "Structured-only checkpoints use input_modalities=['structured']."
            )

    class FusionObservationEncoder(ObservationEncoder):
        """Fuse structured (+ optional future visual) embeddings by concatenation + linear."""

        def __init__(
            self,
            structured: StructuredObservationEncoder,
            visual: VisualObservationEncoder | None = None,
            embed_dim: int = 96,
        ) -> None:
            in_dim = structured.embed_dim + (visual.embed_dim if visual is not None else 0)
            super().__init__(input_dim=in_dim, embed_dim=embed_dim)
            self.structured = structured
            self.visual = visual
            self.fuse = nn.Linear(in_dim, embed_dim)

        def forward(self, batch: Any) -> Any:
            # batch may be Tensor (structured only) or dict with keys
            if isinstance(batch, dict):
                s = self.structured(batch["structured"])
                if self.visual is not None and "visual" in batch:
                    v = self.visual(batch["visual"])
                    x = torch.cat([s, v], dim=-1)
                else:
                    x = s
            else:
                x = self.structured(batch)
            return self.fuse(x)

else:  # pragma: no cover

    class ObservationEncoder:  # type: ignore[no-redef]
        pass

    class StructuredObservationEncoder:  # type: ignore[no-redef]
        pass

    class VisualObservationEncoder:  # type: ignore[no-redef]
        pass

    class FusionObservationEncoder:  # type: ignore[no-redef]
        pass


def encoder_metadata(modalities: Sequence[str] | None = None) -> dict[str, Any]:
    return {
        "input_modalities": list(modalities or ["structured"]),
        "visual_implemented": False,
    }
