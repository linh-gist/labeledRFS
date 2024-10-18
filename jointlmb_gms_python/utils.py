import numpy as np
from scipy.linalg import cholesky
from murty import Murty


def kalman_predict_multiple_noloop(model, m, P):
    plength = m.shape[1]

    m_predict = model.F @ m
    P_predict = np.stack([model.Q] * plength) + np.matmul((model.F @ P).transpose(2, 1, 0), model.F.T)
    P_predict = P_predict.transpose(1, 2, 0)

    return m_predict, P_predict

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
    zlength, plength = z.shape[1], m.shape[1]
    if zlength == 0:
        z_gate = []
        return z_gate
    valid_idx = np.empty(0)
    for j in range(plength):
        Sj = model.R + np.linalg.multi_dot([model.H, P[:,:,j], model.H.T])
        Vs = cholesky(Sj)
        inv_sqrt_Sj = np.linalg.inv(Vs)
        nu = z - model.H @ np.tile(m[:, j].reshape(-1, 1), (1, zlength))
        dist = sum(np.square(inv_sqrt_Sj.T @ nu))

        valid_idx = unique_faster(np.append(valid_idx, np.nonzero(dist < gamma)[0]))

    return valid_idx


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


def kalman_update_multiple_noloop(z, model, m, P):
    plength = m.shape[1]
    P = P.transpose(2, 0, 1)
    y = np.stack([z] * plength) - (model.H @ m).T
    y = y[:, :, np.newaxis]
    S = np.stack([model.R] * plength) + np.matmul(model.H @ P, model.H.T)
    invS = np.linalg.inv(S)
    K = np.matmul(P @ model.H.T, invS)

    mn = m + np.squeeze(K @ y, axis=2).T
    Pn = P - np.matmul(K @ S, K.transpose(0, 2, 1))

    yInvSy = np.squeeze(np.matmul(y.transpose(0, 2, 1), invS @ y), axis=(1, 2))
    qz = np.exp(-0.5 * len(z) * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(S)) - 0.5 * yInvSy)

    return qz, mn, Pn.transpose(1, 2, 0)

def kalman_update_multiple(z, model, m, P):
    plength = m.shape[1]

    qz_update = np.zeros(plength)
    m_update = np.zeros((model.x_dim, plength))
    P_update = np.zeros((model.x_dim, model.x_dim, plength))

    for idxp in range(0, plength):
        qz_temp, m_temp, P_temp = kalman_update_single(z, model.H, model.R, m[:, idxp].reshape(-1, 1), P[:, :, idxp])
        qz_update[idxp] = qz_temp
        m_update[:, idxp] = m_temp.flatten()
        P_update[:, :, idxp] = P_temp

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
                     0.5 * np.dot((z - mu).T, iS @ (z - mu)))
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


def gaus_prune(w, x, P, elim_threshold):
    idx = np.nonzero(w > elim_threshold)[0]
    w_new = w[idx]
    x_new = x[:, idx]
    P_new = P[:, :, idx]
    return w_new, x_new, P_new

def gaus_merge(w,x,P,threshold):
    L= len(w)
    x_dim= x.shape[0]
    I= np.arange(L)
    w_new, x_new, P_new = np.empty(0), np.empty((x_dim, 0)), np.empty((x_dim, x_dim, 0))

    if np.count_nonzero(w)==0:
        return w_new,x_new,P_new

    while len(I):
        j = np.argmax(w)
        Ij = np.empty(0, dtype=int)
        iPt = np.linalg.inv(P[:,:,j])

        for i in I:
            val= np.linalg.multi_dot([(x[:,i]-x[:,j]).T, iPt, (x[:,i]-x[:,j])])
            if val <= threshold:
                Ij = np.append(Ij, i)
        w_new_t = sum(w[Ij])
        x_new_t = np.sum(x[:,Ij]*w[Ij], axis=1)
        P_new_t = np.sum(P[:,:,Ij]*w[Ij], axis=2)

        x_new_t = x_new_t/w_new_t
        P_new_t = P_new_t/w_new_t

        w_new = np.append(w_new, w_new_t)
        x_new = np.column_stack((x_new, x_new_t))
        P_new = np.dstack((P_new, P_new_t))

        I= np.setdiff1d(I,Ij)
        w[Ij]= -1

    return w_new,x_new,P_new

def gaus_cap(w, x, P, max_number):
    if len(w) > max_number:
        idx = np.argsort(-w)
        w_new = w[idx[:max_number]]
        w = w_new * (sum(w)/sum(w_new))
        x = x[:, idx[:max_number]]
        P = P[:, :, idx[:max_number]]
    return w, x, P

def esf(Z):
    """
    Calculate elementary symmetric function using Mahler's recursive formula

    cardinality 1: r1 + r2 + .. + rn
    cardinality 2: r1*r2 + r1*3 + ... + r2*3 + ..

    Parameters
    ----------
    Z: array_like
        Input vector

    Returns
    -------
    out: ndarray
    """
    n_z = len(Z)
    if n_z == 0:
        return np.ones(1)

    F = np.zeros((2, n_z))
    i_n = 0
    i_n_minus = 1

    for n in range(n_z):
        F[i_n, 0] = F[i_n_minus, 0] + Z[n]
        for k in range(1, n + 1):
            if k == n:
                F[i_n, k] = Z[n] * F[i_n_minus, k - 1]
            else:
                F[i_n, k] = F[i_n_minus, k] + Z[n] * F[i_n_minus, k - 1]

        i_n, i_n_minus = i_n_minus, i_n

    return np.concatenate((np.ones(1), F[i_n_minus, :]))

if __name__ == '__main__':

    import os
    from scipy.io import savemat

    if os.path.exists('./w.npy'):
        w=np.load('w.npy')
        x = np.load('x.npy')
        P = np.load('P.npy')
    else:
        w = np.random.dirichlet(np.ones(10))
        x = np.random.rand(4, 10)*10
        P = np.random.rand(4,4,10)*10

        savemat('w.mat', {'w':w})
        savemat('x.mat', {'x': x})
        savemat('P.mat', {'P': P})

        np.save('w.npy', w)
        np.save('x.npy', x)
        np.save('P.npy', P)

    w,x,p = gaus_merge(w,x,P, 4)
    # x = load('x.mat');
    # x = x.x;
    # P = load('P.mat');
    # P = P.P;
    # w = load('w.mat');
    # w = w.w;
    # w = reshape(w, [10 1]);
    # [wn, xn, pn] = gaus_merge(w, x, P, 4);

    P0 = np.array([[0.0304592074847086, np.inf, np.inf, np.inf, 7.41858090274813, np.inf, np.inf, np.inf, -0.345108847352739, np.inf, np.inf],
                   [np.inf, 0.0304592074847086, np.inf, np.inf, np.inf, 7.41858090274813, np.inf, np.inf, np.inf, -0.849090957754662, np.inf],
                   [np.inf, np.inf, 0.0304592074847086, np.inf, np.inf, np.inf, 7.41858090274813, np.inf, np.inf, np.inf, 1.64038243547480],
                   [np.inf, np.inf, np.inf, 0.0304592074847086, np.inf, np.inf, np.inf, 7.41858090274813, np.inf, np.inf, np.inf]])
    gibbswrap_jointpredupdt_custom(P0, 1000)