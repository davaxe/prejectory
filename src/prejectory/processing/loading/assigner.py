"""Deterministic assigners used for split routing and grouping."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence


class StatelessWeightedAssigner[T]:
    """Stateless group assigner based on hashing.

    Given a set of groups and optional weights, this class provides a
    deterministic assignment of any combination of integer values to one of the
    groups. The same seed and the same set of values will always yield the same
    group assignment. The weights determine the probability distribution over
    the groups.

    The advantage of this approach is that it does not require maintaining any
    internal state, which allows deterministic group assignment for the same
    set of input values and seed. However, the downside is that it will only
    converge to the specified distribution in the limit of infinitely many
    assignments, and could differ significantly from the specified distribution
    for small numbers of assignments.

    Parameters
    ----------
    groups : Sequence[T]
        A sequence of unique groups to assign the stream into.
    weights : Sequence[float], optional
        A sequence of weights corresponding to each group. If `None` groups are
        treated as equally weighted. The weights will be normalized to sum to 1
    seed : int, optional
        An optional random seed for deterministic group assignment. If `None`, a
        random seed will be generated internally.

    """

    def __init__(
        self,
        groups: Sequence[T],
        weights: Sequence[float] | None = None,
        seed: int | None = None,
    ) -> None:
        self._seed: int = (
            seed if seed is not None else int(np.random.default_rng().integers(0, 2**31 - 1))
        )
        self._groups: list[T]
        self._weights: npt.NDArray[np.float64]
        self._groups, self._weights = _prepare_groups_and_weights(groups, weights)
        self._cumulative_weights: npt.NDArray[np.float64] = np.cumsum(self._weights)
        self._cumulative_weights[-1] = 1.0

    def assign(self, *values: int | str) -> T:
        """Assign a group based on the provided values.

        Given the same seed and the same set of values, this method will always
        return the same assignment.

        Parameters
        ----------
        *values : int
            A set of integer values that will be used to compute a hash for
            group assignment.

        """
        hash_float = _hash_to_unit_interval(self._seed, *values)
        group_index = int(np.searchsorted(self._cumulative_weights, hash_float, side="right"))
        return self._groups[group_index]


def _hash_to_unit_interval(seed: int, *values: int | str) -> float:
    h = hashlib.blake2b(digest_size=16)
    h.update(str(seed).encode("utf-8"))
    h.update(b"|")
    for v in values:
        h.update(repr(v).encode("utf-8"))
        h.update(b"|")
    digest = h.digest()
    x = int.from_bytes(digest[:8], "big", signed=False)
    return x / 2**64


def _prepare_groups_and_weights[T](
    groups: Sequence[T],
    weights: Sequence[float] | None,
) -> tuple[list[T], npt.NDArray[np.float64]]:
    """Validate and normalize assigner groups and weights."""
    normalized_groups = list(dict.fromkeys(groups))
    if len(groups) != len(normalized_groups):
        msg = (
            f"Groups must be unique. Found {len(groups)} elements "
            f"but only {len(normalized_groups)} unique groups."
        )
        raise ValueError(msg)

    if weights is not None and len(weights) != len(normalized_groups):
        msg = (
            f"Number of weights must match number of groups. "
            f"Found {len(weights)} weights but {len(normalized_groups)} groups."
        )
        raise ValueError(msg)

    normalized_weights = (
        np.array(weights, dtype=np.float64)
        if weights is not None
        else np.ones(len(normalized_groups), dtype=np.float64)
    )
    if np.any(normalized_weights < 0):
        msg = "All weights must be non-negative."
        raise ValueError(msg)

    weights_sum = float(normalized_weights.sum())
    if weights_sum == 0:
        msg = f"At least one weight must be greater than zero got {normalized_weights}."
        raise ValueError(msg)

    normalized_weights /= weights_sum
    return normalized_groups, normalized_weights
