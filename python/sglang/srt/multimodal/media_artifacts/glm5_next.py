"""Prompt-independent GLM-5.3 image preprocessing artifacts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch


@dataclass(frozen=True)
class Glm5NextImagePreprocessArtifact:
    """One reusable GLM-5.3 image grid and its prepared patch tensor."""

    content_digest: str
    artifact_key: str
    feature_hash: int
    grid_thw: tuple[int, int, int]
    feature: Optional[torch.Tensor]

    @property
    def has_feature(self) -> bool:
        return self.feature is not None

    def cache_value(self) -> Glm5NextImagePreprocessArtifact:
        """Never retain a CUDA tensor in the bounded CPU cache."""
        if self.feature is None or self.feature.device.type == "cpu":
            return self
        return replace(self, feature=None)

    def cache_size_items(self) -> tuple:
        return (
            self.content_digest,
            self.artifact_key,
            self.feature_hash,
            self.grid_thw,
            self.feature,
        )
