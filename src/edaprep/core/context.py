"""The fit context: everything a transformer needs beyond ``X`` and ``y``.

Passing one object instead of five keyword arguments keeps the transformer signature
stable as the library grows, and gives every transformer the same journal so that the
report describes the whole run rather than a per-transformer fragment.

The context is created at ``fit`` time and carried into ``transform``.  It holds no
data -- only the profile, the config, the shared journal and the seeded RNG -- which is
what lets ``transform`` be a pure function of fitted state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..config import Config
from ..profiling.profiler import ColumnProfile, DatasetProfile
from ..types import ModelFamily
from .journal import Journal

__all__ = ["FitContext"]


@dataclass
class FitContext:
    """Shared state for one pipeline run."""

    config: Config = field(default_factory=Config)
    profile: Optional[DatasetProfile] = None
    target: Optional[str] = None
    journal: Journal = field(default_factory=Journal)
    #: Free-form slot for transformers that must hand information downstream, for
    #: example the datetime expander telling the scaler which columns it created.
    shared: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.config.random_state)

    @property
    def rng(self) -> np.random.Generator:
        """The seeded generator.

        A single generator is shared by the whole run, so a pipeline with the same
        ``random_state`` and the same steps reproduces exactly, while two different
        stochastic steps do not accidentally draw the same numbers.
        """
        return self._rng

    def fresh_rng(self, salt: int = 0) -> np.random.Generator:
        """An independent generator derived from ``random_state``.

        Used where a step must be reproducible *independently* of how many random
        draws earlier steps happened to make -- refitting one transformer in isolation
        must give the same answer as refitting it inside the pipeline.
        """
        seed = self.config.random_state
        if seed is None:
            return np.random.default_rng()
        return np.random.default_rng(int(seed) + int(salt))

    @property
    def model_family(self) -> Optional[ModelFamily]:
        return self.config.model_family

    @property
    def random_state(self) -> Optional[int]:
        return self.config.random_state

    def column_profile(self, name: str) -> Optional[ColumnProfile]:
        if self.profile is None:
            return None
        return self.profile.columns.get(name)

    def __repr__(self) -> str:
        return (
            f"FitContext(target={self.target!r}, "
            f"model_family={self.model_family}, journal={len(self.journal)} entries)"
        )
