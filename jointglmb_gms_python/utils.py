import numpy as np
from scipy.linalg import cholesky
from murty import Murty


def kalman_predict_multiple(model, m, P):
    plength = m.shape[1];

    m_predict = np.zeros(m.shape);
    P_predict = np.zeros(P.shape);

    for idxp in range(0, plength):
        m_temp, P_temp = kalman_predict_single(model.F, model.Q, m[:, idxp], P[:, :, idxp]);
        m_predict[:, idxp] = m_temp;
        P_predict[:, :, idxp] = P_temp;

    return m_predict, P_predict


def kalman_predict_single(F, Q, m, P):
    m_predict = F @ m;
    P_predict = Q + np.linalg.multi_dot([F, P, F.T]);
    return m_predict, P_predict


def gate_meas_gms_idx(z, gamma, model, m, P):
    zlength = z.shape[1];
    if zlength == 0:
        z_gate = [];
        return;

    Sj = model.R + np.linalg.multi_dot([model.H, P, model.H.T]);

    Vs = cholesky(Sj);
    det_Sj = np.prod(np.diagflat(Vs)) ** 2;
    inv_sqrt_Sj = np.linalg.inv(Vs);
    iSj = np.dot(inv_sqrt_Sj, inv_sqrt_Sj.T);
    nu = z - model.H @ np.tile(m, (1, zlength));
    dist = sum(np.square(inv_sqrt_Sj.T @ nu));

    valid_idx = unique_faster(np.nonzero(dist < gamma)[0])

    return valid_idx;


def new_dict_by_index(dict, idxs):
    if len(dict)==0:
        return dict
    temp = {}
    for idx in idxs:
        temp[idx] = dict[idx]
    return temp

def unique_faster(keys):
    keys = np.sort(keys)
    difference = np.diff(np.append(keys, np.nan))
    keys = keys[np.nonzero(difference)[0]]

    return keys


def kalman_update_multiple(z, model, m, P):
    plength = m.shape[1];
    zlength = z.shape[1];

    qz_update = np.zeros(plength, zlength);
    m_update = np.zeros(model.x_dim, plength, zlength);
    P_update = np.zeros(model.x_dim, model.x_dim, plength);

    for idxp in range(0, plength):
        qz_temp, m_temp, P_temp = kalman_update_single(z, model.H, model.R, m[:, idxp], P[:, :, idxp]);
        qz_update[idxp, :] = qz_temp;
        m_update[:, idxp, :] = m_temp;
        P_update[:, :, idxp] = P_temp;

    return qz_update, m_update, P_update


def kalman_update_single(z, H, R, m, P):
    mu = H @ m;
    S = R + np.linalg.multi_dot([H, P, H.T]);
    Vs = np.linalg.cholesky(S);
    det_S = np.prod(np.diag(Vs)) ** 2;
    inv_sqrt_S = np.linalg.inv(Vs);
    iS = inv_sqrt_S @ inv_sqrt_S.T;
    K = np.linalg.multi_dot([P, H.T, iS]);

    z = z.reshape((-1, 1))
    qz_temp = np.exp(-0.5 * len(z) * np.log(2 * np.pi) - 0.5 * np.log(det_S) -
                     0.5 * np.dot((z - mu).T, iS @ (z - mu))).T;
    m_temp = m + K @ (z - mu);
    P_temp = (np.eye(len(P)) - K @ H) @ P;

    return qz_temp, m_temp, P_temp


def sub2ind(array_shape, rows, cols):
    return rows + array_shape[0] * cols


def gibbswrap_jointpredupdt_custom(P0, m):
    n1 = P0.shape[0];

    if m == 0:
        m = 1 # return at least one solution

    assignments = np.zeros((m, n1));
    costs = np.zeros(m);

    currsoln = np.arange(n1, 2 * n1);  # use all missed detections as initial solution
    assignments[0, :] = currsoln;
    costs[0] = sum(P0.flatten('F')[sub2ind(P0.shape, np.arange(0, n1), currsoln)]);
    for sol in range(1, m):
        for var in range(0, n1):
            tempsamp = np.exp(-P0[var, :]);  # grab row of costs for current association variable
            # lock out current and previous iteration step assignments except for the one in question
            tempsamp[np.delete(currsoln, var)] = 0;
            idxold = np.nonzero(tempsamp > 0)[0];
            tempsamp = tempsamp[idxold];
            currsoln[var] = np.digitize(np.random.rand(1), np.concatenate(([0], np.cumsum(tempsamp) / sum(tempsamp))));
            currsoln[var] = idxold[currsoln[var]-1];
        assignments[sol, :] = currsoln;
        costs[sol] = sum(P0.flatten('F')[sub2ind(P0.shape, np.arange(0, n1), currsoln)]);
    C, I, _ = np.unique(assignments, return_index=True, return_inverse=True, axis=0);
    assignments = C;
    costs = costs[I];

    return assignments, costs

def murty(P0, m):
    n1 = P0.shape[0];
    mgen = Murty(P0)
    assignments = np.zeros((m, n1));
    costs = np.zeros(m);
    sol = 0
    # for cost, assignment in murty(C_ext):
    for sol in range(0, m):
        ok, cost_m, assignment_m = mgen.draw()
        if (not ok):
            break
        assignments[sol, :] = assignment_m;
        costs[sol] = cost_m
    C, I, _ = np.unique(assignments[:sol,:], return_index=True, return_inverse=True, axis=0);
    assignments = C;
    costs = costs[I];
    return assignments, costs

def intersect_mtlb(a, b):
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]


def setxor_mtlb(a, b):
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    c = np.setxor1d(a1, b1)
    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]


if __name__ == '__main__':
    P0 = np.array([[0.0304592074847086, np.inf, np.inf, np.inf, 7.41858090274813, np.inf, np.inf, np.inf, -0.345108847352739, np.inf, np.inf],
                   [np.inf, 0.0304592074847086, np.inf, np.inf, np.inf, 7.41858090274813, np.inf, np.inf, np.inf, -0.849090957754662, np.inf],
                   [np.inf, np.inf, 0.0304592074847086, np.inf, np.inf, np.inf, 7.41858090274813, np.inf, np.inf, np.inf, 1.64038243547480],
                   [np.inf, np.inf, np.inf, 0.0304592074847086, np.inf, np.inf, np.inf, 7.41858090274813, np.inf, np.inf, np.inf]])
    gibbswrap_jointpredupdt_custom(P0, 1000)