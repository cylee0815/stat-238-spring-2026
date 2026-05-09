"""Thompson-sampling decision rule on a fitted hierarchical posterior.

A :class:`ThompsonPolicy` is a typed view onto the posterior ``beta`` block
returned by :func:`project.sampler.gibbs_gaussian` (or
:func:`project.sampler_t.gibbs_student_t`): chains and draws are flattened
into a single posterior-sample axis, optionally thinned. Two action-selection
modes are supported:

- *Per-step Thompson* (``draw_idx=None``, default for ``thompson_action``):
  draw a fresh posterior index every call. This is the textbook bandit-style
  Thompson sampling.
- *Per-episode posterior sampling* (``draw_idx`` given, default for
  :func:`project.rollout.simulate_path`): hold one posterior draw fixed for
  the entire rollout. This is the Osband-style PSRL regime; it is the
  headline regime for the project, with the per-step variant exposed as a
  sweep knob in notebook 07.

The action set is implicit in the trace: ``policy.n_actions`` equals
``trace['beta'].shape[2]``. Action *labels* (e.g., the env's ``{-1, 0, +1}``)
are out of scope here; this module returns integer action ids in
``range(n_actions)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from project.features import FEATURE_COLS


@dataclass(frozen=True)
class ThompsonPolicy:
    """Posterior coefficients reshaped for fast Thompson sampling.

    Attributes
    ----------
    beta_draws
        Float array of shape ``(n_draws_total, n_actions, p)`` where
        ``n_draws_total = n_chains * n_draws // thin``. Stored
        C-contiguous for fast row indexing.
    feature_cols
        The feature ordering this policy expects at decision time. Carried
        only as a label; downstream code is responsible for handing in
        ``x`` rows in this order.
    n_actions
        Convenience copy of ``beta_draws.shape[1]``.
    """

    beta_draws: np.ndarray
    feature_cols: tuple[str, ...]
    n_actions: int

    @property
    def n_draws_total(self) -> int:
        return int(self.beta_draws.shape[0])

    @property
    def p(self) -> int:
        return int(self.beta_draws.shape[2])


def policy_from_trace(
    trace: dict[str, Any],
    *,
    feature_cols: tuple[str, ...] = FEATURE_COLS,
    thin: int = 1,
) -> ThompsonPolicy:
    """Flatten the chains x draws axis of a trace into a Thompson policy.

    Parameters
    ----------
    trace
        Dict with key ``"beta"`` of shape ``(n_chains, n_draws, A, p)``.
    feature_cols
        Tuple of feature names. Must have length ``p`` (sanity check
        against silent column-order bugs at decision time).
    thin
        Keep every ``thin``-th flattened draw. ``thin=1`` keeps everything;
        ``thin=2`` halves the draws. Must be >= 1.
    """
    if thin < 1:
        raise ValueError(f"thin must be >= 1, got {thin}")
    beta = np.asarray(trace["beta"])
    if beta.ndim != 4:
        raise ValueError(f"trace['beta'] must be 4-D (n_chains, n_draws, A, p), got shape {beta.shape}")
    n_chains, n_draws, A, p = beta.shape
    if len(feature_cols) != p:
        raise ValueError(
            f"feature_cols has length {len(feature_cols)} but trace beta has p={p}"
        )

    flat = beta.reshape(n_chains * n_draws, A, p)
    if thin > 1:
        flat = flat[::thin]
    return ThompsonPolicy(
        beta_draws=np.ascontiguousarray(flat, dtype=float),
        feature_cols=tuple(feature_cols),
        n_actions=int(A),
    )


def thompson_action(
    policy: ThompsonPolicy,
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    draw_idx: int | None = None,
) -> int:
    """Select an action from one Thompson posterior sample.

    Parameters
    ----------
    policy
        Fitted :class:`ThompsonPolicy`.
    x
        State features of shape ``(p,)`` matching ``policy.feature_cols``.
    rng
        Used only when ``draw_idx is None``; otherwise ignored.
    draw_idx
        If given, this exact posterior index is used (per-episode PSRL).
        If ``None``, a fresh index is drawn from ``rng`` (per-step Thompson).

    Returns
    -------
    int
        ``argmax_a (x @ beta_draws[idx, a])`` in ``range(n_actions)``.
    """
    if draw_idx is None:
        idx = int(rng.integers(0, policy.n_draws_total))
    else:
        idx = int(draw_idx)
        if not 0 <= idx < policy.n_draws_total:
            raise IndexError(
                f"draw_idx={idx} out of range [0, {policy.n_draws_total})"
            )
    q = policy.beta_draws[idx] @ x          # (A,)
    return int(np.argmax(q))


def posterior_q(policy: ThompsonPolicy, x: np.ndarray) -> np.ndarray:
    """Posterior of ``Q(s, a)`` at one state ``x``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_draws_total, n_actions)``. Each row is one posterior
        draw of the action-value vector at this state. Use for credible-
        interval bands and posterior-density plots in notebook 04.
    """
    x = np.asarray(x, dtype=float)
    return policy.beta_draws @ x            # (D, A, p) @ (p,) -> (D, A)
