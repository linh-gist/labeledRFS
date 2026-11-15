"""Multi-object tracking metrics.

Set-level (per frame):   GOSPA (+ localisation / missed / false split), OSPA, cardinality error
Track-level (sequence):  OSPA(2) (Beard et al. 2020; the repo ships compute_ospa2.m),
                         CLEAR-MOT (MOTA, MOTP, ID switches, fragmentations),
                         IDF1 / IDP / IDR, MT / PT / ML,
                         track_recall    = share of truth tracks covered >= `cover` of their life
                         track_precision = share of estimated tracks (len >= `min_len`) that sit on a
                                           single truth track >= 50% of their life
Detector (per frame):    Pd and false alarms per frame, before tracking.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def gospa(X, Y, c=10.0, p=2, alpha=2):
    """X truth (n,2), Y estimates (m,2). Returns dict(total, loc, missed, false, n_missed, n_false)."""
    n, m = len(X), len(Y)
    if n == 0 and m == 0:
        return dict(total=0.0, loc=0.0, missed=0.0, false=0.0, n_missed=0, n_false=0)
    if n == 0 or m == 0:
        nm, nf = n, m
        val = (c ** p / alpha * (n + m)) ** (1 / p)
        return dict(total=val, loc=0.0, missed=(c ** p / alpha * nm) ** (1 / p),
                    false=(c ** p / alpha * nf) ** (1 / p), n_missed=nm, n_false=nf)
    D = np.minimum(cdist(X, Y), c)
    r, cc = linear_sum_assignment(D ** p)
    ok = D[r, cc] < c
    loc = np.sum(D[r[ok], cc[ok]] ** p)
    nm = n - ok.sum()
    nf = m - ok.sum()
    tot = loc + c ** p / alpha * (nm + nf)
    return dict(total=tot ** (1 / p), loc=loc ** (1 / p), missed=(c ** p / alpha * nm) ** (1 / p),
                false=(c ** p / alpha * nf) ** (1 / p), n_missed=int(nm), n_false=int(nf))


def ospa(X, Y, c=10.0, p=2):
    n, m = len(X), len(Y)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return float(c)
    D = np.minimum(cdist(X, Y), c) ** p
    r, cc = linear_sum_assignment(D)
    return float(((D[r, cc].sum() + c ** p * abs(m - n)) / max(m, n)) ** (1 / p))


def _tracks(seq_ids, seq_pos):
    """per-frame (ids, pos) -> {id: {k: pos}}"""
    T = {}
    for k, (ids, pos) in enumerate(zip(seq_ids, seq_pos)):
        for i, x in zip(ids, pos):
            T.setdefault(int(i), {})[k] = x
    return T


def ospa2(gt_ids, gt_pos, est_ids, est_pos, c=10.0, p=1, window=10):
    """OSPA(2) on sliding windows ending at each frame. Returns array (K,)."""
    K = len(gt_ids)
    G, E = _tracks(gt_ids, gt_pos), _tracks(est_ids, est_pos)
    out = np.zeros(K)
    for k in range(K):
        ks = range(max(0, k - window + 1), k + 1)
        g = [t for t in G.values() if any(j in t for j in ks)]
        e = [t for t in E.values() if any(j in t for j in ks)]
        n, m = len(g), len(e)
        if n == 0 and m == 0:
            continue
        if n == 0 or m == 0:
            out[k] = c
            continue
        D = np.zeros((n, m))
        for a, tg in enumerate(g):
            for b, te in enumerate(e):
                s, cnt = 0.0, 0
                for j in ks:
                    ig, ie = j in tg, j in te
                    if ig or ie:
                        cnt += 1
                        s += min(c, np.linalg.norm(tg[j] - te[j])) ** p if (ig and ie) else c ** p
                D[a, b] = (s / cnt) ** (1 / p) if cnt else 0.0
        D = np.minimum(D, c) ** p
        r, cc = linear_sum_assignment(D)
        out[k] = ((D[r, cc].sum() + c ** p * abs(m - n)) / max(m, n)) ** (1 / p)
    return out


def clear_mot(gt_ids, gt_pos, est_ids, est_pos, thr=5.0, cover=0.8, min_len=5):
    """CLEAR-MOT + identity + track-level coverage metrics with a distance threshold `thr` (px)."""
    K = len(gt_ids)
    prev = {}                      # gt id -> est id at last match
    tracked = {}                   # gt id -> list of bools per frame present
    fn = fp = idsw = matches = 0
    dsum = 0.0
    n_gt = sum(len(i) for i in gt_ids)
    n_est = sum(len(i) for i in est_ids)
    pair_count = {}                # (g, e) -> frames within thr (for IDF1)
    est_on = {}                    # e -> {g: frames matched}
    est_len = {}
    frag = 0
    last_state = {}
    for k in range(K):
        gi, gp, ei, ep = gt_ids[k], gt_pos[k], est_ids[k], est_pos[k]
        for e in ei:
            est_len[int(e)] = est_len.get(int(e), 0) + 1
        if len(gi) and len(ei):
            D = cdist(gp, ep)
            for a in range(len(gi)):
                for b in range(len(ei)):
                    if D[a, b] <= thr:
                        key = (int(gi[a]), int(ei[b]))
                        pair_count[key] = pair_count.get(key, 0) + 1
            C = np.where(D <= thr, D, 1e6)
            # keep previous correspondences when still valid (CLEAR-MOT rule)
            for a, g in enumerate(gi):
                if int(g) in prev:
                    hit = np.nonzero(ei == prev[int(g)])[0]
                    if len(hit) and D[a, hit[0]] <= thr:
                        C[a, hit[0]] = -1.0
            r, cc = linear_sum_assignment(C)
            ok = C[r, cc] < 1e6
            r, cc = r[ok], cc[ok]
        else:
            r = cc = np.zeros(0, int)
        matched_g = set()
        for a, b in zip(r, cc):
            g, e = int(gi[a]), int(ei[b])
            if g in prev and prev[g] != e:
                idsw += 1
            prev[g] = e
            matched_g.add(g)
            matches += 1
            dsum += np.linalg.norm(gp[a] - ep[b])
            est_on.setdefault(e, {}).setdefault(g, 0)
            est_on[e][g] += 1
        fn += len(gi) - len(r)
        fp += len(ei) - len(r)
        for g in gi:
            g = int(g)
            st = g in matched_g
            tracked.setdefault(g, []).append(st)
            if st and last_state.get(g) is False and any(tracked[g][:-1]):
                frag += 1
            last_state[g] = st

    cov = {g: np.mean(v) for g, v in tracked.items()}
    n_tracks = max(len(cov), 1)
    mt = sum(v >= cover for v in cov.values())
    ml = sum(v <= 0.2 for v in cov.values())
    long_est = [e for e, L in est_len.items() if L >= min_len]
    good_est = [e for e in long_est if est_on.get(e) and max(est_on[e].values()) >= 0.5 * est_len[e]]

    # IDF1: one-to-one gt<->est identity matching that maximises matched frames
    gts = sorted(cov)
    ests = sorted(est_len)
    idtp = 0
    if gts and ests:
        M = np.zeros((len(gts), len(ests)))
        gix = {g: i for i, g in enumerate(gts)}
        eix = {e: i for i, e in enumerate(ests)}
        for (g, e), v in pair_count.items():
            M[gix[g], eix[e]] = v
        r, cc = linear_sum_assignment(-M)
        idtp = M[r, cc].sum()
    return dict(
        MOTA=1 - (fn + fp + idsw) / max(n_gt, 1),
        MOTP=dsum / max(matches, 1),
        IDF1=2 * idtp / max(n_gt + n_est, 1),
        IDP=idtp / max(n_est, 1),
        IDR=idtp / max(n_gt, 1),
        ID_switches=idsw,
        fragmentations=frag,
        FN=fn, FP=fp,
        det_recall=matches / max(n_gt, 1),
        det_precision=matches / max(n_est, 1),
        track_recall=mt / n_tracks,
        track_precision=len(good_est) / max(len(long_est), 1),
        MT=mt, PT=len(cov) - mt - ml, ML=ml,
        n_gt_tracks=len(cov), n_est_tracks=len(est_len),
    )


def detector_stats(gt_pos, Z, r=3.0):
    """Per-frame detection probability and false alarms (detections farther than r from any target)."""
    pd, fa = [], []
    for gp, z in zip(gt_pos, Z):
        zt = z.T
        if len(gp) and len(zt):
            D = cdist(gp, zt)
            pd.append(np.mean(D.min(axis=1) <= r))
            fa.append(int(np.sum(D.min(axis=0) > r)))
        else:
            pd.append(np.nan if len(gp) == 0 else 0.0)
            fa.append(len(zt))
    return np.array(pd), np.array(fa)


def evaluate(gt_ids, gt_pos, est_ids, est_pos, c=10.0, thr=5.0, ospa2_window=10):
    per = [gospa(g, e, c=c) for g, e in zip(gt_pos, est_pos)]
    frame = dict(
        gospa=np.array([d["total"] for d in per]),
        gospa_loc=np.array([d["loc"] for d in per]),
        gospa_missed=np.array([d["missed"] for d in per]),
        gospa_false=np.array([d["false"] for d in per]),
        ospa=np.array([ospa(g, e, c=c) for g, e in zip(gt_pos, est_pos)]),
        ospa2=ospa2(gt_ids, gt_pos, est_ids, est_pos, c=c, window=ospa2_window),
        card_true=np.array([len(g) for g in gt_pos]),
        card_est=np.array([len(e) for e in est_pos]),
    )
    summ = dict(GOSPA=frame["gospa"].mean(), GOSPA_loc=frame["gospa_loc"].mean(),
                GOSPA_missed=frame["gospa_missed"].mean(), GOSPA_false=frame["gospa_false"].mean(),
                OSPA=frame["ospa"].mean(), OSPA2=frame["ospa2"].mean(),
                card_err=np.mean(np.abs(frame["card_est"] - frame["card_true"])))
    summ.update(clear_mot(gt_ids, gt_pos, est_ids, est_pos, thr=thr))
    return summ, frame
