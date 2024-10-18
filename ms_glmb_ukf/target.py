from copy import deepcopy
import numpy as np
from scipy.linalg import cholesky
from scipy.linalg import block_diag


def ut(m, P, alpha, kappa):
    n_x = len(m)
    lambda_ = alpha ** 2 * (n_x + kappa) - n_x
    Psqrtm = cholesky((n_x + lambda_) * P).T
    temp = np.zeros((n_x, 2 * n_x + 1))
    temp[:, 1:n_x + 1], temp[:, n_x + 1:2 * n_x + 1] = -Psqrtm, Psqrtm
    X = np.tile(m, (1, 2 * n_x + 1)) + temp
    w = 0.5 * np.ones((2 * n_x + 1, 1))
    w[0] = lambda_
    w = w / (n_x + lambda_)
    return X, w


def gen_msobservation_fn(model, Xorg, W, q):
    if len(Xorg) == 0:
        return []
    X = np.copy(Xorg)
    X[[6, 7, 8], :] = np.exp((X[[6, 7, 8], :]))
    bbs_noiseless = np.zeros((4, X.shape[1]))
    for i in range(X.shape[1]):
        ellipsoid_c = np.array([X[0, i], X[2, i], X[4, i]])  # xc, yc, zc
        rx, ry, hh = X[6, i], X[7, i], X[8, i]  # half length radius (x, y, z)
        # Quadric general equation 0 = Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz + J
        # Q = [A D/2 E/2 G/2;
        #      D/2 B F/2 H/2;
        #      E/2 F/2 C I/2;
        #      G/2 H/2 I/2 J];
        A, B, C = 1 / (rx ** 2), 1 / (ry ** 2), 1 / (hh ** 2)  # calculations for A, B, C
        D, E, F = 0, 0, 0  # calculations for D, E, F, no rotation (axis-aligned) means D, E, F = 0
        # calculations for G, H, I, J
        PSD = np.diag([A, B, C])
        eig_vals, right_eig = np.linalg.eig(PSD)  # [V,D] = eig(A), right eigenvectors, so that A*V = V*D
        temp_ellip_c = np.dot(right_eig.T, ellipsoid_c)
        ggs = (-2 * np.multiply(temp_ellip_c, eig_vals))
        desired = np.dot(ggs, right_eig.T)
        G, H, I = desired[0], desired[1], desired[2]
        J_temp = np.sum(np.divide(np.power(ggs.T, 2), 4 * eig_vals))  # or sum(eig_vals_vec.*(-temp_ellip_c).^2)
        J = -1 + (J_temp)
        Q = np.array([[A, D / 2, E / 2, G / 2],
                      [D / 2, B, F / 2, H / 2],
                      [E / 2, F / 2, C, I / 2],
                      [G / 2, H / 2, I / 2, J]])  # 4x4 matrix
        C_t = np.linalg.multi_dot([model.cam_mat[q], np.linalg.inv(Q), model.cam_mat[q].T])
        CI = np.linalg.inv(C_t)  # 3x3 matrix
        C_strip = CI[0:2, 0:2]  # [C(1,1:2);C(2,1:2)]; % 2x2 matrix
        eig_vals, right_eig = np.linalg.eig(C_strip)
        x_and_y_vec = 2 * CI[0:2, 2]  # np.array([[2. * CI[0, 2], 2. * CI[1, 2]]])  # extrack D and E
        x_and_y_vec_transformed = np.dot(x_and_y_vec, right_eig)
        h_temp = np.divide(x_and_y_vec_transformed, eig_vals) / 2
        h_temp_squared = np.multiply(eig_vals, np.power(h_temp, 2))
        h = -1 * h_temp
        ellipse_c = np.dot(right_eig, h)
        offset = -np.sum(h_temp_squared) + CI[2, 2]
        if (-offset / eig_vals[0] > 0) and (-offset / eig_vals[1] > 0):
            uu = right_eig[:, 0] * np.sqrt(-offset / eig_vals[0])
            vv = right_eig[:, 1] * np.sqrt(-offset / eig_vals[1])
            e = np.sqrt(np.multiply(uu, uu) + np.multiply(vv, vv))
            bbox = np.vstack((ellipse_c - e, ellipse_c + e)).T
            tl0 = np.amin(bbox[0, :])  # top_left0
            tl1 = np.amin(bbox[1, :])  # top_left1
            br0 = np.amax(bbox[0, :])  # bottm_right0
            br1 = np.amax(bbox[1, :])  # bottm_right1
            bbs_noiseless[:, i] = np.array([tl0, tl1, np.log((br0 - tl0)), np.log((br1 - tl1))])
        else:
            # top_left = [1 1];
            bottm_right = [model.imagesize[0], model.imagesize[1]]
            bbs_noiseless[:, i] = np.array([1, 1, np.log((bottm_right[0] - 1)), np.log((bottm_right[1] - 1))])

    return bbs_noiseless + W  # bounding measurement


class Target:
    # track table for GLMB (cell array of structs for individual tracks)
    # (1) r: existence probability
    # (2) Gaussian Mixture w (weight), m (mean), P (covariance matrix)
    # (3) Label: birth time & index of target at birth time step
    # (4) gatemeas: indexes gating measurement (using  Chi-squared distribution)
    # (5) ah: association history
    def __init__(self, m, P, prob_birth, label, model):
        self.m = m
        self.P = P
        self.w = 1  # weights of Gaussians for birth track
        self.r = prob_birth
        self.P_S = model.P_S
        self.l = label  # track label
        self.gatemeas = [np.array([]) for i in range(model.N_sensors)]
        max_cpmt = 100
        self.ah = np.zeros(max_cpmt, dtype=int)
        self.ah_idx = 0

    def predict(self, model):
        # this is to offset the normal mean because of lognormal multiplicative noise.
        offset = np.zeros((model.x_dim, 1))
        offset[[6, 7, 8]] = np.array([[model.n_mu[0]], [model.n_mu[0]], [model.n_mu[1]]])
        m_per_mode = self.m + offset
        self.m = np.dot(model.F, m_per_mode)
        self.P = model.Q + np.linalg.multi_dot([model.F, self.P, model.F.T])

    def ukf_update_per_sensor(self, z, model, s, alpha, kappa, beta):
        m_hold = self.m[[0, 2, 4]]
        ch1 = m_hold[0] > model.XMAX[0] and m_hold[0] < model.XMAX[1]
        ch2 = m_hold[1] > model.YMAX[0] and m_hold[1] < model.YMAX[1]
        ch3 = m_hold[2] > model.ZMAX[0] and m_hold[2] < model.ZMAX[1]
        if not (ch1 and ch2 and ch3):
            return np.log(np.spacing(1))  # qz_temp
        m, P = np.append(self.m, np.zeros((model.z_dim[s], 1)), axis=0), block_diag(*[self.P, model.R[s]])
        X_ukf, u = ut(m, P, alpha, kappa)
        temp = np.array([[0], [0], [model.meas_n_mu[s, 0]], [model.meas_n_mu[s, 1]]])
        X_ukf[model.x_dim:model.x_dim + model.z_dim[s], :] = X_ukf[model.x_dim:model.x_dim + model.z_dim[s], :] + temp
        Z_pred = gen_msobservation_fn(model, X_ukf[0:model.x_dim, :],
                                      X_ukf[model.x_dim:model.x_dim + model.z_dim[s], :], s)
        eta = np.dot(Z_pred, u)
        S_temp = Z_pred - np.tile(eta, (1, len(u)))
        u[0] = u[0] + (1 - alpha ** 2 + beta)
        S = np.linalg.multi_dot([S_temp, np.diagflat(u), S_temp.T])
        Vs = cholesky(S)
        det_S = np.prod(np.diag(Vs)) ** 2
        inv_sqrt_S = np.linalg.inv(Vs)
        iS = np.dot(inv_sqrt_S, inv_sqrt_S.T)
        # G_temp = X_ukf[0:model.x_dim, :] - np.tile(self.m, (1, len(u)))
        # G = np.linalg.multi_dot([G_temp, np.diagflat(u), S_temp.T])
        # K = np.dot(G, iS)
        z_eta = z - eta
        qz_temp = -0.5 * (z.shape[0] * np.log(2 * np.pi) + np.log(det_S) + np.linalg.multi_dot([z_eta.T, iS, z_eta]))

        qz_temp = np.exp(qz_temp)
        # m_temp = self.m + np.dot(K, z_eta)
        # P_temp = self.P - np.linalg.multi_dot([G, iS, G.T])

        return qz_temp

    def ukf_msjointupdate(self, Z, k, nestmeasidxs, model, alpha, kappa, beta):
        slogidxs = nestmeasidxs > 0
        if not np.any(slogidxs):
            return 1, self
        stacked_R = []
        stacked_z = np.array([])
        nestmeasidxs = nestmeasidxs - 1  # restore original measurement index 0-|Z|
        s_multi = 0
        for idx, logidxs in enumerate(slogidxs):
            if logidxs:
                stacked_R.append(model.R[idx])
                stacked_z = np.append(stacked_z, Z[(k, idx)][:, nestmeasidxs[idx]])
                s_multi += model.z_dim[idx]
        stacked_z = stacked_z[:, np.newaxis]
        stacked_R = block_diag(*stacked_R)
        m, P = np.append(self.m, np.zeros((s_multi, 1)), axis=0), block_diag(*[self.P, stacked_R])
        X_ukf, u = ut(m, P, alpha, kappa)
        start_indx = model.x_dim
        Z_pred = np.zeros((4 * sum(slogidxs), X_ukf.shape[1]))
        z_idx = 0
        for idx, logidxs in enumerate(slogidxs):
            if logidxs:
                end_indx = start_indx + model.z_dim[idx]
                temp = np.array([[0], [0], [model.meas_n_mu[idx, 0]], [model.meas_n_mu[idx, 1]]])
                X_ukf[start_indx:end_indx, :] = X_ukf[start_indx:end_indx, :] + temp
                Z_temp = gen_msobservation_fn(model, X_ukf[0:model.x_dim, :], X_ukf[start_indx:end_indx, :], idx)
                start_indx = end_indx
                Z_pred[4 * z_idx:4 * (z_idx + 1), :] = Z_temp
                z_idx += 1
        eta = np.dot(Z_pred, u)
        S_temp = Z_pred - np.tile(eta, (1, len(u)))
        u[0] = u[0] + (1 - alpha ** 2 + beta)
        S = np.linalg.multi_dot([S_temp, np.diagflat(u), S_temp.T])
        Vs = cholesky(S)
        det_S = np.prod(np.diag(Vs)) ** 2
        inv_sqrt_S = np.linalg.inv(Vs)
        iS = np.dot(inv_sqrt_S, inv_sqrt_S.T)
        G_temp = X_ukf[0:model.x_dim, :] - np.tile(self.m, (1, len(u)))
        G = np.linalg.multi_dot([G_temp, np.diagflat(u), S_temp.T])
        K = np.dot(G, iS)
        z_eta = stacked_z - eta
        qz_temp = -0.5 * (stacked_z.shape[0] * np.log(2 * np.pi) + np.log(det_S) +
                          np.linalg.multi_dot([z_eta.T, iS, z_eta]))  # log domain
        qz_temp = np.exp(qz_temp)
        m_temp = self.m + np.dot(K, z_eta)
        P_temp = self.P - np.linalg.multi_dot([G, iS, G.T])
        tt_update = deepcopy(self)
        tt_update.m = m_temp
        tt_update.P = P_temp

        return qz_temp, tt_update

    def gate_msmeas_ukf(self, model, gamma, Zz, k, alpha, kappa, beta):
        for s in range(model.N_sensors):
            z = Zz[k, s]
            zlength = z.shape[1]
            if zlength == 0:
                self.gatemeas[s] = np.array([])
            m, P = np.append(self.m, np.zeros((model.z_dim[s], 1)), axis=0), block_diag(*[self.P, model.R[s]])
            X_ukf, u = ut(m, P, alpha, kappa)
            Z_pred = gen_msobservation_fn(model, X_ukf[0:model.x_dim, :],
                                          X_ukf[model.x_dim:model.x_dim + model.z_dim[s], :], s)
            eta = np.dot(Z_pred, u)
            Sj_temp = Z_pred - np.tile(eta, (1, len(u)))
            u[0] = u[0] + (1 - alpha ** 2 + beta)
            Sj = np.linalg.multi_dot([Sj_temp, np.diagflat(u), Sj_temp.T])
            Vs = cholesky(Sj)
            # det_Sj = np.prod(np.diag(Vs)) ** 2
            inv_sqrt_Sj = np.linalg.inv(Vs)
            # iSj = np.dot(inv_sqrt_Sj, inv_sqrt_Sj.T)
            m_z = gen_msobservation_fn(model, self.m, np.zeros((model.D[s].shape[1], 1)), s)
            nu = z - np.tile(m_z, (1, zlength))
            dist = sum(np.power(np.dot(inv_sqrt_Sj.T, nu), 2))
            self.gatemeas[s] = np.nonzero(dist < gamma[s])[0]

    def not_gating(self, model, Zz, k):
        for s in range(model.N_sensors):
            self.gatemeas[s] = np.arange(0, Zz[k, s].shape[1])
    #  END
