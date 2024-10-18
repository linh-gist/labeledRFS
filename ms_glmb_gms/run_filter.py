from copy import deepcopy

import cv2
from scipy.stats.distributions import chi2
from gen_model import model
import numpy as np
from cpputils import gibbs_jointpredupdt
from plot_results import plot_truth_meas
from utils import gate_meas_gms_idx, kalman_update_multiple, unique_faster, \
    gibbswrap_jointpredupdt_custom, intersect_mtlb, setxor_mtlb, kalman_update_single, murty, kalman_predict_multiple, \
    kalman_predict_single, new_dict_by_index, kalman_update_multiple_2, ndsub2ind, kalman_msjointupdate_multiple
from scipy.special import logsumexp
from scipy.linalg import cholesky, block_diag
from gibbs import gibbs_multisensor_approx_cheap, AdaptiveBirth


class filter:
    # filter parameters
    def __init__(self, model):
        self.H_upd = 3000  # requested number of updated components/hypotheses
        self.H_max = 3000  # cap on number of posterior components/hypotheses
        self.hyp_threshold = 1e-5  # pruning threshold for components/hypotheses

        self.L_max = 100  # limit on number of Gaussians in each track - not implemented yet
        self.elim_threshold = 1e-5  # pruning threshold for Gaussians in each track - not implemented yet
        self.merge_threshold = 4  # merging threshold for Gaussians in each track - not implemented yet

        self.P_G = 0.9999999  # gate size in percentage
        self.gamma = chi2.ppf(self.P_G, model.z_dim)  # inv chi^2 dn gamma value
        self.gate_flag = 1  # gating on or off 1/0

        self.run_flag = 'disp'  # 'disp' or 'silence' for on the fly output


class Estimate:
    def __init__(self):
        self.X = {}
        self.N = {}
        self.L = {}


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
        mtemp_predict, Ptemp_predict = kalman_predict_single(model.F, model.Q, self.m, self.P)
        self.m = mtemp_predict
        self.P = Ptemp_predict

    def update2(self, model, meas, s):
        # tt_update = deepcopy(self)
        # qz_temp, _, _ = kalman_update_multiple_2(meas, model, s, self.m[:, np.newaxis], self.P[:, :, np.newaxis])
        qz_temp, _, _ = kalman_update_single(meas, model.H[s], model.R[s], self.m, self.P)
        # w_temp = np.multiply(qz_temp, self.w) + np.spacing(1)
        return qz_temp

    def msjointupdate(self, Z, k, nestmeasidxs, model):
        slogidxs = nestmeasidxs > 0
        if not np.any(slogidxs):
            return 1, self
        stacked_H = np.concatenate(model.H[slogidxs], axis=0)
        stacked_R = []
        stacked_z = np.array([])
        nestmeasidxs = nestmeasidxs - 1  # restore original measurement index 0-|Z|
        for idx, logidxs in enumerate(slogidxs):
            if logidxs:
                stacked_R.append(model.R[idx])
                stacked_z = np.append(stacked_z, Z[(k, idx)][:, nestmeasidxs[idx]])
        stacked_R = block_diag(*stacked_R)
        qz_temp, m_temp, P_temp = kalman_update_single(stacked_z, stacked_H, stacked_R, self.m, self.P)
        tt_update = deepcopy(self)
        tt_update.m = m_temp
        tt_update.P = P_temp

        return qz_temp, tt_update

    def gating(self, model, gamma, Zz, k):
        for s in range(model.N_sensors):
            z = Zz[k, s]
            zlength = z.shape[1]
            if zlength == 0:
                self.gatemeas[s] = np.array([])
            Sj = model.R[s] + np.linalg.multi_dot([model.H[s], self.P, model.H[s].T])
            Vs = cholesky(Sj)
            # det_Sj = np.prod(np.diagflat(Vs)) ** 2
            inv_sqrt_Sj = np.linalg.inv(Vs)
            # iSj = np.dot(inv_sqrt_Sj, inv_sqrt_Sj.T)
            nu = z - model.H[s] @ np.tile(self.m, (1, zlength))
            dist = sum(np.square(inv_sqrt_Sj.T @ nu))

            self.gatemeas[s] = unique_faster(np.nonzero(dist < gamma[s])[0])

    def not_gating(self, model, Zz, k):
        for s in range(model.N_sensors):
            self.gatemeas[s] = np.arange(0, Zz[k, s].shape[1])
    #  END


class GLMB:  # delta GLMB
    def __init__(self, model):
        # initial Numpy Data type for target
        self.x_dim = 4

        self.glmb_update_tt = []  # 1, track table for GLMB
        self.glmb_update_w = np.array([1])  # 2, vector of GLMB component/hypothesis weights
        self.glmb_update_I = [np.array([], dtype=int)]  # 3, cell of GLMB component/hypothesis labels
        self.glmb_update_n = np.array([0])  # 4, vector of GLMB component/hypothesis cardinalities
        self.glmb_update_cdn = np.array([1])  # 5, cardinality distribution of GLMB
        np.set_printoptions(linewidth=2048)  # use in clean_predict() to convert array2string

        self.tt_birth = []
        self.AdaptiveBirth = AdaptiveBirth()
        mu0 = np.zeros(6)
        P0 = np.power(np.diagflat([2000, 50, 2000, 50, 2000, 50]), 2)
        # num_sensors, F, Q, PD, lambda_c, pdf_c, R,  mu0, P0, tau_rU, H_gibbs, num_miss, r_bmax
        self.AdaptiveBirth.init_parameters(model.N_sensors, model.F, model.Q, model.P_D[0], model.lambda_c[0],
                                           model.pdf_c[0], model.R[0], mu0, P0,
                                           0.9, 1000, 0, model.r_birth[0])
        self.est = Estimate()

    def msjointpredictupdate(self, model, filter, meas, k):
        if k == 0:  # create birth tracks
            meas_ru, meas_Z = [], []
            for s in range(model.N_sensors):
                temp = 0.99 * np.ones(meas.Z[(k, s)].shape[1] + 1)
                temp[0] = 0.01  # for miss-detection
                meas_ru.append(temp)
                meas_Z.append(meas.Z[(k, s)])
            m, P, r_b = self.AdaptiveBirth.sample_adaptive_birth(meas_ru, meas_Z)
            for idx, P_idx in enumerate(P):
                tt = Target(m[:, idx][:, np.newaxis], P_idx, r_b[idx], np.array([0, idx]), model)
                self.tt_birth.append(tt)
        # create surviving tracks - via time prediction (single target CK)
        for tt in self.glmb_update_tt:
            tt.predict(model)

        # create predicted tracks - concatenation of birth and survival
        glmb_predict_tt = self.tt_birth + self.glmb_update_tt  # copy track table back to GLMB struct

        # gating by tracks
        if filter.gate_flag:
            for tt in glmb_predict_tt:
                tt.gating(model, filter.gamma, meas.Z, k)
        else:
            for tabidx in range(0, len(glmb_predict_tt)):
                tt.not_gating(model, meas.Z, k)

        # precalculation loop for average survival/death probabilities
        avps = np.array([tt.r for tt in self.tt_birth] +
                        [tt.P_S for tt in self.glmb_update_tt])[:, np.newaxis]
        avqs = 1 - avps

        # precalculation loop for average detection/missed probabilities
        avpd = np.zeros((len(glmb_predict_tt), model.N_sensors))
        for tabidx in range(0, len(glmb_predict_tt)):
            for s in range(model.N_sensors):
                avpd[tabidx, s] = model.P_D[s]
        avqd = 1 - avpd

        # create updated tracks (single target Bayes update)
        m = np.zeros(model.N_sensors, dtype=int)
        for s in range(model.N_sensors):
            m[s] = meas.Z[k, s].shape[1]  # number of measurements

        # nested for loop over all predicted tracks and sensors - slow way
        # Kalman updates on the same prior recalculate all quantities
        # extra state for not detected for each sensor (meas "0" in pos 1)
        allcostc = [np.zeros((len(glmb_predict_tt), 1 + m[s])) for s in range(model.N_sensors)]
        # extra state for not detected for each sensor (meas "0" in pos 1)
        jointcostc = [np.zeros((len(glmb_predict_tt), 1 + m[s])) for s in range(model.N_sensors)]
        # posterior probability of target survival (after measurement updates)
        avpp = np.zeros((len(glmb_predict_tt), model.N_sensors))
        for s in range(model.N_sensors):
            # allcostc[s] # extra state for not detected for each sensor (meas "0" in pos 1)
            # jointcostc[s] # extra state for not detected for each sensor (meas "0" in pos 1)
            for tabidx, tt in enumerate(glmb_predict_tt):
                for emm in tt.gatemeas[s]:
                    w_temp = tt.update2(model, meas.Z[k, s][:, emm, np.newaxis], s)  # unnormalized updated weights
                    allcostc[s][tabidx, 1 + emm] = sum(w_temp)  # predictive likelihood

            cij_d = avps * avqd[:, s][:, np.newaxis]  # survived and missed detection
            cij_m = np.tile(avps * avpd[:, s][:, np.newaxis], (1, m[s]))  # survived and detected targets
            jointcostc[s] = np.column_stack((cij_d, cij_m)) * allcostc[s] / (model.lambda_c[s] * model.pdf_c[s])
            jointcostc[s][:, 0] = cij_d[:, 0]
            avpp[:, s] = np.sum(jointcostc[s], axis=1)

        # gated measurement index matrix
        gatemeasidxs = [-1 * np.ones((len(glmb_predict_tt), m[s]), dtype="int") for s in range(model.N_sensors)]
        for s in range(model.N_sensors):
            for tabidx, tt in enumerate(glmb_predict_tt):
                gatemeasidxs[s][tabidx, tt.gatemeas[s]] = tt.gatemeas[s]

        # component updates
        runidx = 0
        glmb_nextupdate_w = np.zeros(filter.H_max * 2)
        glmb_nextupdate_I = []
        glmb_nextupdate_n = np.zeros(filter.H_max * 2, dtype=int)
        # use to normalize assign_prob using glmb_nextupdate_w, first raw for "measurement" missed detection
        assign_meas = [np.zeros((m[s] + 1, filter.H_max * 2), dtype=int) for s in range(model.N_sensors)]
        cpreds = len(glmb_predict_tt)
        nbirths = len(self.tt_birth)
        hypoth_num = np.rint(filter.H_upd * np.sqrt(self.glmb_update_w) / sum(np.sqrt(self.glmb_update_w))).astype(int)
        neglog_jointcostm = [jointcostc[s] + np.spacing(0) for s in range(model.N_sensors)]  # negative log cost

        tt_update_parent = np.array([], dtype="int")
        tt_update_currah = np.empty((0, model.N_sensors), dtype="int")
        tt_update_linidx = np.array([], dtype="int")

        for pidx in range(0, len(self.glmb_update_w)):
            # calculate best updated hypotheses/components
            # indices of all births and existing tracks  for current component
            tindices = np.concatenate((np.arange(0, nbirths), nbirths + self.glmb_update_I[pidx]))
            mindices = []
            for s in range(model.N_sensors):
                # union indices of gated measurements for corresponding tracks
                ms_indices = np.unique(gatemeasidxs[s][tindices, :])
                if -1 in ms_indices:
                    ms_indices = ms_indices[1:]
                mindices.append(np.insert(1 + ms_indices, 0, 0).astype("int"))
            costc = []
            for s in range(model.N_sensors):
                take_rows = neglog_jointcostm[s][tindices]
                costc.append(take_rows[:, mindices[s]])
            dcost = avqs[tindices]  # death cost
            scost = np.prod(avpp[tindices, :], axis=1)[:, np.newaxis]  # posterior survival cost
            dprob = dcost / (dcost + scost)

            uasses = gibbs_multisensor_approx_cheap(dprob, costc, hypoth_num[pidx])
            # uasses = np.array(uasses, dtype="f8")
            # uasses[uasses < 0] = -np.inf  # set not born/track deaths to -inf assignment

            # generate corrresponding jointly predicted/updated hypotheses/components
            local_avpdm = avpd[tindices, :]
            local_avqdm = avqd[tindices, :]
            repeated_lambda_c = np.tile(model.lambda_c.T, (len(tindices), 1))
            repeated_pdf_c = np.tile(model.pdf_c.T, (len(tindices), 1))
            for hidx in range(len(uasses)):
                update_hypcmp_tmp = uasses[hidx]
                off_idx = update_hypcmp_tmp[:, 0] < 0
                aug_idx = np.column_stack((tindices, update_hypcmp_tmp))  # [tindices, 1 + update_hypcmp_tmp]
                mis_idx = update_hypcmp_tmp == 0
                det_idx = update_hypcmp_tmp > 0
                update_hypcmp_idx = np.zeros(len(off_idx))
                update_hypcmp_idx[off_idx] = -np.inf
                update_hypcmp_idx[~off_idx] = ndsub2ind(np.insert(1 + m, 0, cpreds), aug_idx[~off_idx, :])
                num_trk = sum(update_hypcmp_idx >= 0)

                sum_temp = -sum(model.lambda_c) + m[np.newaxis, :] @ np.log(model.lambda_c * model.pdf_c)
                sum_temp += sum(np.log(avps[tindices[~off_idx]])) + sum(np.log(avqs[tindices[off_idx]]))
                sum_temp += sum(np.log(local_avpdm[det_idx])) + sum(np.log(local_avqdm[mis_idx]))
                sum_temp -= sum(np.log(repeated_lambda_c[det_idx] * repeated_pdf_c[det_idx]))
                glmb_nextupdate_w[runidx] = sum_temp + np.log(self.glmb_update_w[pidx])  # hypothesis/component weight

                # Get measurement index from uasses (make sure minus 1 from [mindices+1])
                for s in range(model.N_sensors):
                    miss_idx = update_hypcmp_tmp[mis_idx[:, s], s].astype(int)  # first raw for missed detection
                    meas_idx = update_hypcmp_tmp[det_idx[:, s], s].astype(int)
                    assign_meas[s][miss_idx, runidx] = 1
                    assign_meas[s][meas_idx, runidx] = 1  # Setting index of measurements associate with a track

                if num_trk > 0:
                    # hypothesis/component tracks (via indices to track table)
                    glmb_nextupdate_I.append(np.arange(len(tt_update_parent), len(tt_update_parent) + num_trk))
                else:
                    glmb_nextupdate_I.append([])
                glmb_nextupdate_n[runidx] = num_trk  # hypothesis/component cardinality
                runidx = runidx + 1

                tt_update_parent = np.append(tt_update_parent, tindices[~off_idx])
                tt_update_currah = np.row_stack((tt_update_currah, update_hypcmp_tmp[~off_idx, :].astype("int")))
                tt_update_linidx = np.append(tt_update_linidx, update_hypcmp_idx[update_hypcmp_idx >= 0].astype("int"))
        # END

        # component updates via posterior weight correction (including generation of track table)
        ttU_allkey, ttU_oldidx, ttU_newidx = np.unique(tt_update_linidx, return_index=True, return_inverse=True)
        tt_update_msqz = np.zeros((len(ttU_allkey), 1))
        tt_update = []
        for tabidx in range(len(ttU_allkey)):
            oldidx = ttU_oldidx[tabidx]
            preidx = tt_update_parent[oldidx]
            meascomb = tt_update_currah[oldidx, :]

            # kalman update for this track and all joint measurements
            qz_temp, tt = glmb_predict_tt[preidx].msjointupdate(meas.Z, k, meascomb, model)
            tt_update_msqz[tabidx] = qz_temp

            tt_update.append(tt)
        # END

        for pidx in range(runidx):
            glmb_nextupdate_I[pidx] = ttU_newidx[glmb_nextupdate_I[pidx]]
            glmb_nextupdate_w[pidx] = glmb_nextupdate_w[pidx] + sum(np.log(tt_update_msqz[glmb_nextupdate_I[pidx]]))

        # normalize weights
        glmb_nextupdate_w = glmb_nextupdate_w[:runidx]
        glmb_nextupdate_n = glmb_nextupdate_n[:runidx]
        glmb_nextupdate_w = np.exp(glmb_nextupdate_w - logsumexp(glmb_nextupdate_w))

        # Multi-sensor Joint Adaptive Birth Sampler for Labeled Random Finite Set Tracking
        meas_ru, meas_Z = [], []
        for s in range(model.N_sensors):
            rA = assign_meas[s][:, :runidx] @ glmb_nextupdate_w  # adaptive birth weight for each measurement
            rA = np.clip(rA, np.spacing(0), 1 - np.spacing(0))
            meas_ru.append(1 - rA)  # rU
            meas_Z.append(meas.Z[(k, s)])
        m, P, r_b = self.AdaptiveBirth.sample_adaptive_birth(meas_ru, meas_Z)
        self.tt_birth = []
        for idx, P_idx in enumerate(P):
            tt = Target(m[:, idx][:, np.newaxis], P_idx, r_b[idx], np.array([k + 1, idx]), model)
            self.tt_birth.append(tt)

        # extract cardinality distribution
        glmb_nextupdate_cdn = np.zeros(max(glmb_nextupdate_n) + 1)
        for card in range(0, max(glmb_nextupdate_n) + 1):
            # extract probability of n targets
            glmb_nextupdate_cdn[card] = sum(glmb_nextupdate_w[glmb_nextupdate_n == card])
        # END

        # extract cardinality distribution
        glmb_nextupdate_cdn = np.zeros(max(glmb_nextupdate_n) + 1)
        for card in range(0, max(glmb_nextupdate_n) + 1):
            # extract probability of n targets
            glmb_nextupdate_cdn[card] = sum(glmb_nextupdate_w[glmb_nextupdate_n == card])

        # copy glmb update to the next time step
        self.glmb_update_tt = tt_update  # 1, copy track table back to GLMB struct
        self.glmb_update_w = glmb_nextupdate_w  # 2
        self.glmb_update_I = glmb_nextupdate_I  # 3
        self.glmb_update_n = glmb_nextupdate_n  # 4
        self.glmb_update_cdn = glmb_nextupdate_cdn  # 5

        # remove duplicate entries and clean track table
        self.clean_predict()
        self.clean_update()

    def clean_predict(self):
        # hash label sets, find unique ones, merge all duplicates
        glmb_raw_hash = np.zeros(len(self.glmb_update_w), dtype=np.dtype('<U2048'))
        for hidx in range(0, len(self.glmb_update_w)):
            hash_str = np.array2string(np.sort(self.glmb_update_I[hidx]), separator='*')[1:-1]
            glmb_raw_hash[hidx] = hash_str

        cu, _, ic = np.unique(glmb_raw_hash, return_index=True, return_inverse=True, axis=0)

        glmb_temp_w = np.zeros((len(cu)))
        glmb_temp_I = [np.array([]) for i in range(0, len(ic))]
        glmb_temp_n = np.zeros((len(cu)), dtype=int)
        for hidx in range(0, len(ic)):
            glmb_temp_w[ic[hidx]] = glmb_temp_w[ic[hidx]] + self.glmb_update_w[hidx]
            glmb_temp_I[ic[hidx]] = self.glmb_update_I[hidx]
            glmb_temp_n[ic[hidx]] = self.glmb_update_n[hidx]

        self.glmb_update_w = glmb_temp_w  # 2
        self.glmb_update_I = glmb_temp_I  # 3
        self.glmb_update_n = glmb_temp_n  # 4

    def clean_update(self):
        # flag used tracks
        usedindicator = np.zeros(len(self.glmb_update_tt), dtype=int)
        for hidx in range(0, len(self.glmb_update_w)):
            usedindicator[self.glmb_update_I[hidx]] = usedindicator[self.glmb_update_I[hidx]] + 1
        trackcount = sum(usedindicator > 0)

        # remove unused tracks and reindex existing hypotheses/components
        newindices = np.zeros(len(self.glmb_update_tt), dtype=int)
        newindices[usedindicator > 0] = np.arange(0, trackcount)
        glmb_clean_tt = [self.glmb_update_tt[i] for i, indicator in enumerate(usedindicator) if indicator > 0]
        self.glmb_update_tt = glmb_clean_tt  # 1

        glmb_clean_I = []
        for hidx in range(0, len(self.glmb_update_w)):
            glmb_clean_I.append(newindices[self.glmb_update_I[hidx]])

        self.glmb_update_I = glmb_clean_I  # 3

    def prune(self, filter):
        # prune components with weights lower than specified threshold
        idxkeep = np.nonzero(self.glmb_update_w > filter.hyp_threshold)[0]
        glmb_out_w = self.glmb_update_w[idxkeep]
        glmb_out_I = [self.glmb_update_I[i] for i in idxkeep]
        glmb_out_n = self.glmb_update_n[idxkeep]

        glmb_out_w = glmb_out_w / sum(glmb_out_w)
        glmb_out_cdn = np.zeros((max(glmb_out_n) + 1))
        for card in range(0, np.max(glmb_out_n) + 1):
            glmb_out_cdn[card] = sum(glmb_out_w[glmb_out_n == card])

        self.glmb_update_w = glmb_out_w  # 2
        self.glmb_update_I = glmb_out_I  # 3
        self.glmb_update_n = glmb_out_n  # 4
        self.glmb_update_cdn = glmb_out_cdn  # 5

    def cap(self, filter):
        # cap total number of components to specified maximum
        if len(self.glmb_update_w) > filter.H_max:
            idxsort = np.argsort(-self.glmb_update_w)
            idxkeep = idxsort[0:filter.H_max]
            glmb_out_w = self.glmb_update_w[idxkeep]
            glmb_out_I = [self.glmb_update_I[i] for i in idxkeep]
            glmb_out_n = self.glmb_update_n[idxkeep]

            glmb_out_w = glmb_out_w / sum(glmb_out_w)
            glmb_out_cdn = np.zeros(max(glmb_out_n) + 1)
            for card in range(0, max(glmb_out_n) + 1):
                glmb_out_cdn[card] = sum(glmb_out_w[glmb_out_n == card])

            self.glmb_update_w = glmb_out_w  # 2
            self.glmb_update_I = glmb_out_I  # 3
            self.glmb_update_n = glmb_out_n  # 4
            self.glmb_update_cdn = glmb_out_cdn  # 5

    def extract_estimates(self, model, meas):
        # extract estimates via recursive estimator, where
        # trajectories are extracted via association history, and
        # track continuity is guaranteed with a non-trivial estimator

        # extract MAP cardinality and corresponding highest weighted component
        N = np.argmax(self.glmb_update_cdn)
        X = np.zeros((model.x_dim, N))
        L = np.zeros((2, N), dtype=int)

        idxcmp = np.argmax(np.multiply(self.glmb_update_w, (self.glmb_update_n == N).astype(int)))
        for n in range(0, N):
            idxtrk = np.argmax(self.glmb_update_tt[self.glmb_update_I[idxcmp][n]].w)
            idxcmp_I = self.glmb_update_I[idxcmp][n]
            X[:, n] = self.glmb_update_tt[idxcmp_I].m[:, idxtrk]
            L[:, n] = self.glmb_update_tt[idxcmp_I].l

        return X, N, L

    def display_diaginfo(self, k, est, filter, H_predict, H_posterior, H_prune, H_cap):
        if filter.run_flag is not 'silence':
            print(' time= ', str(k),
                  ' #eap cdn=', "{:10.4f}".format(np.arange(0, (len(self.glmb_update_cdn))) @ self.glmb_update_cdn),
                  ' #var cdn=', "{:10.4f}".format(np.arange(0, (len(self.glmb_update_cdn)))**2 @ self.glmb_update_cdn -
                                    (np.arange(0, (len(self.glmb_update_cdn))) @ self.glmb_update_cdn) ** 2),
                  ' #est card=', "{:10.4f}".format(est[1]),
                  ' #comp pred=', "{:10.4f}".format(H_predict),
                  ' #comp post=', "{:10.4f}".format(H_posterior),
                  ' #comp updt=', "{:10.4f}".format(H_cap),
                  ' #trax updt=', "{:10.4f}".format(len(self.glmb_update_tt)))

    def run(self, model, filter, meas):
        for k in range(0, meas.K):
            # joint prediction and update
            self.msjointpredictupdate(model, filter, meas, k)
            H_posterior = len(self.glmb_update_w)

            # pruning and truncation
            self.prune(filter)
            H_prune = len(self.glmb_update_w)
            self.cap(filter)
            H_cap = len(self.glmb_update_w)
            self.clean_update()

            # state estimation and display diagnostics
            est = self.extract_estimates(model, meas)
            self.est.X[k], self.est.N[k], self.est.L[k] = est
            self.display_diaginfo(k, est, filter, H_posterior, H_posterior, H_prune, H_cap)
        return self.est
    # END
