"""Pure-Python stand-ins for the compiled helpers the repo's `*_fast` trackers import.

The repo expects two compiled extensions:
  * ``murty``  (github.com/JohnPekl/murty, C++, not on PyPI)  -> ``Murty(C).draw()``
  * ``gibbs``  (cpp_gibbs, pybind11 + Eigen, needs MSVC)    -> ``gibbs_jointpredupdt(C, n)``

If the real modules are importable they are used untouched. Otherwise these
implementations are registered under the same names so the repo code runs as-is.
The algorithms are the same (Murty's k-best ranked assignment; Gibbs sampling of
Vo et al. 2017, Alg. 1, initialised with the optimal assignment), but they run in
Python, so absolute latencies are higher than the C++ builds would give.
"""
from __future__ import annotations

import heapq
import sys
import types

import numpy as np
from scipy.optimize import linear_sum_assignment

_BIG = 1e12


def _finite(C):
    C = np.asarray(C, dtype=float)
    return np.where(np.isfinite(C), C, _BIG)


def _lsa(C):
    """Optimal assignment of every row (rows <= cols). Returns (cols, cost) or None."""
    if C.shape[0] == 0:
        return np.zeros(0, dtype=int), 0.0
    r, c = linear_sum_assignment(C)
    vals = C[r, c]
    if np.any(vals >= _BIG / 2):
        return None
    return c, float(vals.sum())


class Murty:
    """k-best assignments in increasing cost order (Murty 1968), API of JohnPekl/murty."""

    def __init__(self, C):
        self.C = _finite(C)
        self.n = self.C.shape[0]
        self._heap = []
        self._cnt = 0
        sol = self._solve({}, frozenset())
        if sol is not None:
            self._push(sol, {}, frozenset())

    def _push(self, sol, forced, excluded):
        assign, cost = sol
        heapq.heappush(self._heap, (cost, self._cnt, assign, forced, excluded))
        self._cnt += 1

    def _solve(self, forced, excluded):
        n = self.n
        free_rows = [i for i in range(n) if i not in forced]
        used_cols = set(forced.values())
        free_cols = [j for j in range(self.C.shape[1]) if j not in used_cols]
        sub = self.C[np.ix_(free_rows, free_cols)].copy() if free_rows else np.zeros((0, 0))
        if free_rows:
            col_pos = {j: p for p, j in enumerate(free_cols)}
            row_pos = {i: p for p, i in enumerate(free_rows)}
            for (i, j) in excluded:
                if i in row_pos and j in col_pos:
                    sub[row_pos[i], col_pos[j]] = _BIG
        res = _lsa(sub)
        if res is None:
            return None
        cols, cost = res
        assign = np.empty(n, dtype=int)
        for i, j in forced.items():
            assign[i] = j
            cost += self.C[i, j]
        for p, i in enumerate(free_rows):
            assign[i] = free_cols[cols[p]]
        return assign, cost

    def draw(self):
        if not self._heap:
            return False, 0.0, None
        cost, _, assign, forced, excluded = heapq.heappop(self._heap)
        f = dict(forced)
        for i in range(self.n):
            if i in forced:
                continue
            ex = excluded | {(i, int(assign[i]))}
            sol = self._solve(f, ex)
            if sol is not None:
                self._push(sol, dict(f), ex)
            f[i] = int(assign[i])
        return True, cost, assign.astype(float)


def gibbs_jointpredupdt(P0, m, rng=None):
    """Gibbs sampler over joint (birth/survival/detection) assignments.

    P0 : (n, 2n+M) negative-log cost matrix, inf = forbidden.
    Returns unique assignments (float array, column index per row) and their costs.
    """
    rng = np.random.default_rng() if rng is None else rng
    P0 = np.asarray(P0, dtype=float)
    n1 = P0.shape[0]
    if n1 == 0:
        return np.zeros((1, 0)), np.zeros(1)
    m = max(int(m), 1)
    init = _lsa(_finite(P0))
    curr = init[0].copy() if init is not None else np.arange(n1, 2 * n1)
    # row-wise normalised likelihoods (inf cost -> 0 weight)
    rowmin = np.min(np.where(np.isfinite(P0), P0, np.inf), axis=1, keepdims=True)
    W = np.exp(-(P0 - rowmin))
    W[~np.isfinite(P0)] = 0.0
    samples = np.empty((m, n1), dtype=int)
    samples[0] = curr
    u = rng.random((m, n1))
    for s in range(1, m):
        for i in range(n1):
            w = W[i].copy()
            keep = w[curr[i]]
            w[curr] = 0.0          # columns taken by other rows
            w[curr[i]] = keep
            cs = np.cumsum(w)
            if cs[-1] > 0:
                curr[i] = int(np.searchsorted(cs, u[s, i] * cs[-1], side="right"))
        samples[s] = curr
    uniq = np.unique(samples, axis=0)
    costs = P0[np.arange(n1)[None, :], uniq].sum(axis=1)
    return uniq.astype(float), costs


def murty_kbest(P0, m):
    """Same contract as gibbs_jointpredupdt but with exact k-best (Murty)."""
    P0 = np.asarray(P0, dtype=float)
    n1 = P0.shape[0]
    if n1 == 0:
        return np.zeros((1, 0)), np.zeros(1)
    gen = Murty(P0)
    A, c = [], []
    for _ in range(max(int(m), 1)):
        ok, cost, a = gen.draw()
        if not ok:
            break
        A.append(a)
        c.append(cost)
    if not A:
        return np.zeros((0, n1)), np.zeros(0)
    return np.array(A, dtype=float), np.array(c)


def install():
    """Register shim modules for any compiled dependency that is missing."""
    status = {}
    try:
        import murty  # noqa: F401
        status["murty"] = "compiled"
    except ImportError:
        mod = types.ModuleType("murty")
        mod.Murty = Murty
        sys.modules["murty"] = mod
        status["murty"] = "python-shim"
    try:
        import gibbs  # noqa: F401
        status["gibbs"] = "compiled"
    except ImportError:
        mod = types.ModuleType("gibbs")
        mod.gibbs_jointpredupdt = gibbs_jointpredupdt
        sys.modules["gibbs"] = mod
        status["gibbs"] = "python-shim"
    return status
