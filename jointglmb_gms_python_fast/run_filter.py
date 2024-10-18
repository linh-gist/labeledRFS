from copy import deepcopy

import cv2
from scipy.stats.distributions import chi2
from gen_model import model
import numpy as np
from gibbs import gibbs_jointpredupdt
from plot_results import plot_truth_meas
from utils import gate_meas_gms_idx, kalman_update_multiple, unique_faster, \
    gibbswrap_jointpredupdt_custom, intersect_mtlb, setxor_mtlb, kalman_update_single, murty, kalman_predict_multiple, \
    kalman_predict_single, new_dict_by_index
from scipy.special import logsumexp


class filter:
    # filter parameters
    def __init__(self, model):
        self.H_upd = 1000  # requested number of updated components/hypotheses
        self.H_max = 1000  # cap on number of posterior components/hypotheses
        self.hyp_threshold = 1e-15  # pruning threshold for components/hypotheses

        self.L_max = 100  # limit on number of Gaussians in each track - not implemented yet
        self.elim_threshold = 1e-5  # pruning threshold for Gaussians in each track - not implemented yet
        self.merge_threshold = 4  # merging threshold for Gaussians in each track - not implemented yet

        self.P_G = 0.9999999  # gate size in percentage
        self.gamma = chi2.ppf(self.P_G, model.z_dim)  # inv chi^2 dn gamma value
        self.gate_flag = 1  # gating on or off 1/0

        self.run_flag = 'disp'  # 'disp' or 'silence' for on the fly output


class est:
    def __init__(self, meas):
        self.X = []
        self.N = np.zeros(meas.K)
        self.L = []
        self.T = []
        self.M = 0
        self.J = np.empty((2, 0), dtype=int)
        self.H = np.empty(0)

class Target:
    # track table for GLMB (cell array of structs for individual tracks)
    # (1) r: existence probability
    # (2) Gaussian Mixture w (weight), m (mean), P (covariance matrix)
    # (3) Label: birth time & index of target at birth time step
    # (4) gatemeas: indexes gating measurement (using  Chi-squared distribution)
    # (5) ah: association history
    def __init__(self, z, prob_birth, label, model):
        self.m = np.array([[z[0]], [0], [z[1]], [0]])
        self.P = model.P_birth[0]  # covs of Gaussians for birth track
        self.w = 1  # weights of Gaussians for birth track
        self.r = prob_birth
        self.l = label  # track label
        max_cpmt = 100
        self.ah = np.zeros(max_cpmt, dtype=int)
        self.ah_idx = 0

    def predict(self, model):
        mtemp_predict, Ptemp_predict = kalman_predict_single(model.F, model.Q, self.m, self.P)
        self.m = mtemp_predict
        self.P = Ptemp_predict

    def update(self, model, meas, emm):
        tt_update = deepcopy(self)
        qz_temp, m_temp, P_temp = kalman_update_single(meas, model.H, model.R, self.m, self.P)
        w_temp = np.multiply(qz_temp, self.w) + np.spacing(1)

        cost_update = sum(w_temp)
        tt_update.w = w_temp / sum(w_temp)
        tt_update.m = m_temp
        tt_update.P = P_temp
        tt_update.ah[tt_update.ah_idx] = emm
        tt_update.ah_idx += 1
        return cost_update, tt_update

    def gating(self, model, filter_gama, meas_z):
        self.gatemeas = gate_meas_gms_idx(meas_z, filter_gama, model, self.m, self.P)
    #  END

class GLMB:  # delta GLMB
    def __init__(self):
        # initial Numpy Data type for target
        self.x_dim = 4

        self.glmb_update_tt = []  # 1, track table for GLMB
        self.glmb_update_w = np.array([1])  # 2, vector of GLMB component/hypothesis weights
        self.glmb_update_I = [np.array([], dtype=int)]  # 3, cell of GLMB component/hypothesis labels
        self.glmb_update_n = np.array([0])  # 4, vector of GLMB component/hypothesis cardinalities
        self.glmb_update_cdn = np.array([1])  # 5, cardinality distribution of GLMB

        self.p_birth = 0.001
        self.tt_birth_steps = {}
        # self.tt_birth = np.empty(0, dtype=self.target_dtype)

    def jointpredictupdate(self, model, filter, meas, k):
        # create surviving tracks - via time prediction (single target CK)
        for tt in self.glmb_update_tt:
            tt.predict(model)

        # create predicted tracks - concatenation of birth and survival
        glmb_predict_tt = self.tt_birth + self.glmb_update_tt  # copy track table back to GLMB struct

        # gating by tracks
        if filter.gate_flag:
            for tt in glmb_predict_tt:
                tt.gating(model, filter.gamma, meas.Z[k])
        else:
            for tabidx in range(0, len(glmb_predict_tt)):
                tt.gatemeas = np.arange(0, meas.Z[k].shape[1])

        # precalculation loop for average survival/death probabilities
        avps = np.array([tt.r for tt in self.tt_birth] +
                        [model.P_S for r in range(len(self.glmb_update_tt))])[:, np.newaxis]
        avqs = 1 - avps

        # precalculation loop for average detection/missed probabilities
        avpd = np.zeros((len(glmb_predict_tt), 1))
        for tabidx in range(0, len(glmb_predict_tt)):
            avpd[tabidx] = model.P_D
        avqd = 1 - avpd

        # create updated tracks (single target Bayes update)
        m = meas.Z[k].shape[1]  # number of measurements
        tt_update = deepcopy(glmb_predict_tt) + [[] for i in range((m) * len(glmb_predict_tt))]  # initialize cell array
        # missed detection tracks (legacy tracks)
        for tabidx in range(0, len(glmb_predict_tt)):
            # track association history (updated for missed detection)
            tt_update[tabidx].ah[tt_update[tabidx].ah_idx] = -1
            tt_update[tabidx].ah_idx = tt_update[tabidx].ah_idx + 1
        # measurement updated tracks (all pairs)
        allcostm = np.zeros((len(glmb_predict_tt), m))
        for tabidx, tt in enumerate(glmb_predict_tt):
            for emm in glmb_predict_tt[tabidx].gatemeas:
                stoidx = len(glmb_predict_tt) * (emm + 1) + tabidx
                cost_update, tt_update_gate = tt.update(model, meas.Z[k][:, emm], emm)
                allcostm[tabidx, emm] = cost_update
                tt_update[stoidx] = tt_update_gate
        # joint cost matrix, cij_z is the cost matrix for survived and detected tracks
        cij_z = np.multiply(np.tile(np.multiply(avps, avpd), (1, m)), allcostm) / (model.lambda_c * model.pdf_c)
        jointcostm = np.concatenate((np.diagflat(avqs),
                                     np.diagflat(np.multiply(avps, avqd)),
                                     cij_z), axis=1)
        # gated measurement index matrix
        gatemeasidxs = -1 * np.ones((len(glmb_predict_tt), m), dtype=int)
        for tabidx, tt in enumerate(glmb_predict_tt):
            gatemeasidxs[tabidx, tt.gatemeas] = tt.gatemeas
        gatemeasindc = gatemeasidxs >= 0

        # component updates
        runidx = 0
        glmb_nextupdate_w = np.zeros(filter.H_max * 2)
        glmb_nextupdate_I = []
        glmb_nextupdate_n = np.zeros(filter.H_max * 2, dtype=int)
        self.assign_prob = np.zeros(m)  # adaptive birth weight for each measurement
        assign_meas = np.zeros((m, filter.H_max * 2), dtype=int)  # use to normalize assign_prob using glmb_nextupdate_w
        cpreds = len(glmb_predict_tt)
        nbirths = len(self.tt_birth)
        lselmask = np.zeros((len(glmb_predict_tt), m), dtype=bool)
        hypoth_num = np.rint(filter.H_upd * np.sqrt(self.glmb_update_w) / sum(np.sqrt(self.glmb_update_w))).astype(int)
        neglog_jointcostm = -np.log(jointcostm)  # negative log cost

        for pidx in range(0, len(self.glmb_update_w)):
            # calculate best updated hypotheses/components
            nexists = len(self.glmb_update_I[pidx])
            ntracks = nbirths + nexists
            # indices of all births and existing tracks  for current component
            tindices = np.concatenate((np.arange(0, nbirths), nbirths + self.glmb_update_I[pidx]))
            lselmask[:] = False
            lselmask[tindices, :] = gatemeasindc[tindices, :]  # logical selection mask to index gating matrices
            # union indices of gated measurements for corresponding tracks
            mindices = np.unique(gatemeasidxs[lselmask])
            # cost matrix - [no_birth/is_death | born/survived+missed | born/survived+detected]
            take_rows = neglog_jointcostm[tindices]
            neglogcostm = take_rows[:, np.concatenate((tindices, cpreds + tindices, 2 * cpreds + mindices))]
            # murty's algo/gibbs sampling to calculate m-best assignment hypotheses/components
            uasses, nlcost = murty(neglogcostm, hypoth_num[pidx])  # output theta, measurement to track association
            uasses = uasses + 1
            uasses[uasses <= ntracks] = -np.inf  # set not born/track deaths to -inf assignment
            uasses[(uasses > ntracks) & (uasses <= 2 * ntracks)] = 0  # set survived+missed to 0 assignment
            # set survived+detected to assignment of measurement index from 1:|Z|
            uasses[uasses > 2 * ntracks] = uasses[uasses > 2 * ntracks] - 2 * ntracks
            # restore original indices of gated measurements
            uasses[uasses > 0] = mindices[uasses[uasses > 0].astype(int) - 1] + 1

            # generate corrresponding jointly predicted/updated hypotheses/components
            len_nlcost = len(nlcost)
            # hypothesis/component weight eqs (20) => (15) => (17) => (4), omega_z
            # Vo Ba-Ngu "An efficient implementation of the generalized labeled multi-Bernoulli filter."
            glmb_nextupdate_w[runidx:runidx + len_nlcost] = -model.lambda_c + m * np.log(
                model.lambda_c * model.pdf_c) + np.log(self.glmb_update_w[pidx]) - nlcost
            # hypothesis/component cardinality
            glmb_nextupdate_n[runidx:runidx + len_nlcost] = np.sum(uasses >= 0, axis=1)
            for hidx in range(len_nlcost):
                update_hypcmp_tmp = uasses[hidx, :]
                update_hypcmp_idx = cpreds * update_hypcmp_tmp + np.concatenate(
                    (np.arange(0, nbirths), nbirths + self.glmb_update_I[pidx])).astype(int)
                # Get measurement index from uasses (make sure minus 1 from [mindices+1])
                uasses_idx = update_hypcmp_tmp[update_hypcmp_tmp > 0].astype(int) - 1
                assign_meas[uasses_idx, runidx] = 1  # Setting index of measurements associate with a track
                # hypothesis/component tracks (via indices to track table)
                glmb_nextupdate_I.append(update_hypcmp_idx[update_hypcmp_idx >= 0].astype(int))
                runidx = runidx + 1

        # normalize weights
        glmb_nextupdate_w = glmb_nextupdate_w[:runidx]
        glmb_nextupdate_n= glmb_nextupdate_n[:runidx]
        glmb_nextupdate_w = np.exp(glmb_nextupdate_w - logsumexp(glmb_nextupdate_w))
        assign_prob = assign_meas[:, :runidx] @ glmb_nextupdate_w
        # create birth tracks
        self.apdative_birth(assign_prob, meas, model, k)

        # extract cardinality distribution
        glmb_nextupdate_cdn = np.zeros(max(glmb_nextupdate_n) + 1)
        for card in range(0, max(glmb_nextupdate_n) + 1):
            glmb_nextupdate_cdn[card] = sum(
                glmb_nextupdate_w[glmb_nextupdate_n == card])  # extract probability of n targets

        # copy glmb update to the next time step
        self.glmb_update_tt = tt_update  # 1, copy track table back to GLMB struct
        self.glmb_update_w = glmb_nextupdate_w  # 2
        self.glmb_update_I = glmb_nextupdate_I  # 3
        self.glmb_update_n = glmb_nextupdate_n  # 4
        self.glmb_update_cdn = glmb_nextupdate_cdn  # 5

        # remove duplicate entries and clean track table
        self.clean_predict()
        self.clean_update()

    def apdative_birth(self, assign_prob, meas, model, k):
        not_assigned_sum = sum(1 - assign_prob)
        lambda_b = 0.1  # Set lambda_b to the mean cardinality of the birth multi-Bernoulli RFS
        self.tt_birth = []
        for idx in range(assign_prob.shape[0]):
            # eq (75) "The Labeled Multi-Bernoulli Filter", Stephan Reuter∗, Ba-Tuong Vo, Ba-Ngu Vo, ...
            prob_birth = np.minimum(self.p_birth, (1 - assign_prob[idx]) / not_assigned_sum * lambda_b)
            tt = Target(meas.Z[k][:, idx], prob_birth, np.array([k + 1, idx]), model)
            self.tt_birth.append(tt)
        self.tt_birth_steps[k + 1] = np.copy(self.tt_birth)

    def clean_predict(self):
        # hash label sets, find unique ones, merge all duplicates
        glmb_raw_hash = np.zeros(len(self.glmb_update_w), dtype=np.dtype('<U32'))
        for hidx in range(0, len(self.glmb_update_w)):
            glmb_raw_hash[hidx] = hash(self.glmb_update_I[hidx].tostring())

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

    def extract_estimates_recursive(self, model, meas, est):
        # extract estimates via recursive estimator, where
        # trajectories are extracted via association history, and
        # track continuity is guaranteed with a non-trivial estimator

        # extract MAP cardinality and corresponding highest weighted component
        mode = np.argmax(self.glmb_update_cdn)
        M = mode
        T = [np.array([], dtype=int) for i in range(0, M)]
        J = np.zeros((2, M), dtype=int)

        idxcmp = np.argmax(np.multiply(self.glmb_update_w, (self.glmb_update_n == M).astype(int)))
        for m in range(0, M):
            idxptr = self.glmb_update_I[idxcmp][m]
            T[m] = self.glmb_update_tt[idxptr].ah[:self.glmb_update_tt[idxptr].ah_idx]
            J[:, m] = self.glmb_update_tt[idxptr].l

        # Until here it is enough to extract tracks and display but if we do not use recursive extraction
        # tracks may not be appeared in some time steps (it is not continuous), we even may miss some tracks.
        # See more: Nguyen, Tran Thien Dat, and Du Yong Kim. "GLMB tracker with partial smoothing." Sensors

        H = np.empty(0)
        for m in range(0, M):
            H = np.append(H, (str(J[0, m]) + '.' + str(J[1, m])))

        # compute dead & updated & new tracks
        _, iio, iis = np.intersect1d(est.H, H, assume_unique=False, return_indices=True)
        _, iid, iin = setxor_mtlb(est.H, H)

        est.M = M
        est.T = [est.T[i] for i in iid]
        est.T += [T[i] for i in iis]
        est.T += [T[i] for i in iin]
        est.J = np.column_stack((est.J[:, iid], J[:, iis], J[:, iin]))
        est.H = np.concatenate((est.H[iid], H[iis], H[iin]))

        # write out estimates in standard format
        est.N = np.zeros(meas.K)
        est.X = {key: np.empty((model.x_dim, 0)) for key in range(0, meas.K)}
        est.L = {key: np.empty((model.z_dim, 0), dtype=int) for key in range(0, meas.K)}
        for t in range(0, len(est.T)):
            ks = est.J[0, t]
            bidx = est.J[1, t]
            tah = est.T[t]

            w = self.tt_birth_steps[ks][bidx].w
            m = self.tt_birth_steps[ks][bidx].m
            P = self.tt_birth_steps[ks][bidx].P
            for u in range(0, len(tah)):
                m, P = kalman_predict_single(model.F, model.Q, m, P)
                k = ks + u
                emm = tah[u]
                if emm >= 0:
                    [qz, m, P] = kalman_update_single(meas.Z[k][:, emm], model.H, model.R, m, P)
                    w = np.multiply(qz, w + np.spacing(1))
                    w = w / sum(w)

                idxtrk = np.argmax(w)
                est.N[k] = est.N[k] + 1
                est.X[k] = np.column_stack((est.X[k], m[:, idxtrk]))
                est.L[k] = np.column_stack((est.L[k], est.J[:, t]))
        return est

    def display_diaginfo(self, k, est, filter, H_predict, H_posterior, H_prune, H_cap):
        if filter.run_flag is not 'silence':
            print(' time= ', str(k),
                  ' #eap cdn=', "{:10.4f}".format(np.arange(0, (len(self.glmb_update_cdn))) @ self.glmb_update_cdn[:]),
                  ' #var cdn=', "{:10.4f}".format(np.arange(0, (len(self.glmb_update_cdn))) ** 2 @ self.glmb_update_cdn[:] -
                                    (np.arange(0, (len(self.glmb_update_cdn))) @ self.glmb_update_cdn[:]) ** 2),
                  ' #est card=', "{:10.4f}".format(est.N[k]),
                  ' #comp pred=', "{:10.4f}".format(H_predict),
                  ' #comp post=', "{:10.4f}".format(H_posterior),
                  ' #comp updt=', "{:10.4f}".format(H_cap),
                  ' #trax updt=', "{:10.4f}".format(len(self.glmb_update_tt)))

    def run(self, model, filter, meas, est):
        self.tt_birth = []
        for idx in range(meas.Z[0].shape[1]):
            tt = Target(meas.Z[0][:, idx], self.p_birth, np.array([0, idx]), model)
            self.tt_birth.append(tt)
        self.tt_birth_steps[0] = deepcopy(self.tt_birth)

        for k in range(0, meas.K):
            # joint prediction and update
            self.jointpredictupdate(model, filter, meas, k)
            H_posterior = len(self.glmb_update_w)

            # pruning and truncation
            self.prune(filter)
            H_prune = len(self.glmb_update_w)
            self.cap(filter)
            H_cap = len(self.glmb_update_w)

            # state estimation and display diagnostics
            est = self.extract_estimates_recursive(model, meas, est)
            self.display_diaginfo(k, est, filter, H_posterior, H_posterior, H_prune, H_cap)

    def plot_tracks(self, model, filter, truth, meas):
        self.tt_birth = []
        for idx in range(meas.Z[0].shape[1]):
            tt = Target(meas.Z[0][:, idx], self.p_birth, np.array([0, idx]), model)
            self.tt_birth.append(tt)
        self.tt_birth_steps[0] = deepcopy(self.tt_birth)

        fig, axs1, axs2 = plot_truth_meas(model, truth, meas)
        for k in range(0, meas.K):
            # joint prediction and update
            self.jointpredictupdate(model, filter, meas, k)
            # pruning and truncation
            self.prune(filter)

            # extract estimates via recursive estimator, where trajectories are extracted via association history, and
            # track continuity is guaranteed with a non-trivial estimator

            # extract MAP cardinality and corresponding highest weighted component
            mode = np.argmax(self.glmb_update_cdn)
            M = mode
            print("Step", k)
            idxcmp = np.argmax(np.multiply(self.glmb_update_w, (self.glmb_update_n == M).astype(int)))
            for m in range(0, M):
                idxptr = self.glmb_update_I[idxcmp][m]

                loc = self.glmb_update_tt[idxptr].m
                label = self.glmb_update_tt[idxptr].l
                label_color = float(hash(str(label[0]) + '.' + str(label[1])) % 256) / 256
                axs1.scatter(k, loc[0], marker='.', s=80, color=label_color * np.ones(3))
                axs2.scatter(k, loc[2], marker='.', s=80, color=label_color * np.ones(3))

                fig.canvas.draw()
                image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                cv2.imshow("Tracking", image)
                cv2.waitKey(1)
    # END
