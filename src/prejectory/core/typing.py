from __future__ import annotations

from pathlib import Path

import numpy as np
from typing_extensions import TypeVar

SourceId = str | int
SourceT = TypeVar("SourceT", default=Path)
FloatDType = type[np.float32 | np.float64]
