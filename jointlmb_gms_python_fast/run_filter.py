from copy import deepcopy
from operator import attrgetter

import cv2
from gibbs import gibbs_jointpredupdt
from scipy.stats.distributions import chi2
import numpy as np

from gen_model import model
from plot_results import plot_truth_meas
from utils import gate_meas_gms_idx, kalman_update_multiple, unique_faster, gaus_cap, \
    gibbswrap_jointpredupdt_custom, murty, kalman_predict_multiple, gaus_prune, gaus_merge, esf
from scipy.special import logsumexp


class filter:
    # filter parameters
    def __init__(self, model):
        self.T_max = 100  # maximum number of tracks
        self.track_threshold = 1e-3  # threshold to prune tracks
        self.H_upd = 1000  # requested number of updated components/hypotheses (for GLMB update)

        self.L_max = 10;  # limit on number of Gaussians in each track - not implemented yet
        self.elim_threshold = 1e-5;  # pruning threshold for Gaussians in each track - not implemented yet
        self.merge_threshold = 4;  # merging threshold for Gaussians in each track - not implemented yet

        self.P_G = 0.9999999;  # gate size in percentage
        self.gamma = chi2.ppf(self.P_G, model.z_dim);  # inv chi^2 dn gamma value
        self.gate_flag = 1;  # gating on or off 1/0

        self.run_flag = 'disp'  # 'disp' or 'silence' for on the fly output


class est:
    def __init__(self, meas):
        self.N = np.zeros(meas.K)
        self.X = {key: np.empty((4, 0)) for key in range(0, meas.K)}
        self.L = {key: np.empty((4, 0), dtype=int) for key in range(0, meas.K)}


class Target:
    # track table for LMB (cell array of structs for individual tracks)
    # (1) r: existence probability
    # (2) Gaussian Mixture w (weight), m (mean), P (covariance matrix)
    # (3) Label: birth time & index of target at birth time step
    # (4) gatemeas: indexes gating measurement (using  Chi-squared distribution)
    def __init__(self):
        self.x_dim = 4
        max_cpmt = 200

        # wg, mg, Pg, ..., store temporary Gaussian mixtures while updating, see 'update_gms'
        self.wg = np.zeros(max_cpmt)
        self.mg = np.zeros((self.x_dim, max_cpmt))
        self.Pg = np.zeros((self.x_dim, self.x_dim, max_cpmt))
        self.idxg = 0

        # store number of Gaussian mixtures before updating, see 'update_gms'
        self.gm_len = 0

    def predict_gms(self, model):
        self.r = model.P_S * self.r
        mtemp_predict, Ptemp_predict = kalman_predict_multiple(model, self.m, self.P)
        self.m = mtemp_predict
        self.P = Ptemp_predict

    def update_gms(self, model, meas):
        self.idxg = 0

        # Gaussian mixtures for misdetection
        length = 1 if np.isscalar(self.w) else len(self.w)
        self.wg[:length] = self.w
        self.mg[:, :length] = self.m
        self.Pg[:, :, :length] = self.P
        self.idxg += length
        self.gm_len = length

        # Kalman update for each Gaussian with each gating measurement
        cost_update = np.zeros(len(self.gatemeas))
        for i, emm in enumerate(self.gatemeas):
            qz_temp, m_temp, P_temp = kalman_update_multiple(meas[:, emm], model, self.m, self.P)
            w_temp = np.multiply(qz_temp, self.w) + np.spacing(1)

            cost_update[i] = sum(w_temp)
            length = len(qz_temp)
            self.wg[self.idxg:self.idxg + length] = w_temp / sum(w_temp)
            self.mg[:, self.idxg:self.idxg + length] = m_temp
            self.Pg[:, :, self.idxg:self.idxg + length] = P_temp
            self.idxg += length

        # Copy values back to each fields
        self.w = self.wg[:self.idxg]
        self.m = self.mg[:, :self.idxg]
        self.P = self.Pg[:, :, :self.idxg]

        return cost_update

    # remove Gaussian mixtures in 'update_gm' that are not in ranked assignment
    def select_gms(self, select_idxs):
        self.w = self.wg[select_idxs]
        self.m = self.mg[:, select_idxs]
        self.P = self.Pg[:, :, select_idxs]

    def finalize_glmb2lmb(self, sums):
        repeat_sums = np.repeat(sums, self.gm_len)
        self.w *= repeat_sums
        self.r = sum(self.w)
        self.w = self.w / self.r
    #  END


class LMB:
    def __init__(self):
        # initial prior
        self.tt_lmb = []
        self.glmb_update_w = np.array([1])  # 2, vector of GLMB component/hypothesis weights

    def jointlmbpredictupdate(self, model, filter, meas, k):
        #  generate birth tracks
        tt_birth = [Target() for i in range(0, len(model.r_birth))]  # initialize cell array
        for tabbidx in range(0, len(model.r_birth)):
            tt_birth[tabbidx].r = model.r_birth[tabbidx]  # birth prob for birth track
            tt_birth[tabbidx].m = model.m_birth[tabbidx]  # means of Gaussians for birth track
            tt_birth[tabbidx].P = model.P_birth[tabbidx][:, :, np.newaxis]  # covs of Gaussians for birth track
            tt_birth[tabbidx].w = model.w_birth[tabbidx]  # weights of Gaussians for birth track
            tt_birth[tabbidx].l = np.array([k, tabbidx])  # track label

        # generate surviving tracks
        for target in self.tt_lmb:
            target.predict_gms(model)

        # create predicted tracks - concatenation of birth and survival
        self.tt_lmb += tt_birth  # copy track table back to GLMB struct
        ntracks = len(self.tt_lmb)

        # gating by tracks
        if filter.gate_flag:
            for tabidx in range(ntracks):
                gate_idx = gate_meas_gms_idx(meas.Z[k], filter.gamma, model, self.tt_lmb[tabidx].m,
                                             self.tt_lmb[tabidx].P)
                self.tt_lmb[tabidx].gatemeas = gate_idx.astype(int)
        else:
            for tabidx in range(ntracks):
                self.tt_lmb[tabidx].gatemeas = range(0, meas.Z[k].shape[1])



        # precalculation loop for average survival/death probabilities
        avps = np.zeros((ntracks, 1))
        for tabidx in range(ntracks):
            avps[tabidx] = self.tt_lmb[tabidx].r
        avqs = 1 - avps

        # precalculation loop for average detection/missed probabilities
        avpd = np.ones((ntracks, 1)) * model.P_D
        avqd = 1 - avpd

        # create updated tracks (single target Bayes update)
        m = meas.Z[k].shape[1]  # number of measurements
        allcostm = np.zeros((ntracks, m))
        for tabidx in range(ntracks):
            cost_update = self.tt_lmb[tabidx].update_gms(model, meas.Z[k])
            allcostm[tabidx, self.tt_lmb[tabidx].gatemeas] = cost_update

        # joint cost matrix, eta_j eq (22) "An Efficient Implementation of the GLMB"
        eta_j = np.multiply(np.multiply(avps, avpd), allcostm) / (model.lambda_c * model.pdf_c)
        jointcostm = np.zeros((ntracks, 2 * ntracks + m))
        jointcostm[:, 0:ntracks] = np.diagflat(avqs)
        jointcostm[:, ntracks:2 * ntracks] = np.diagflat(np.multiply(avps, avqd))
        jointcostm[:, 2 * ntracks:2 * ntracks + m] = eta_j

        # calculate best updated hypotheses/components
        # murty's algo/gibbs sampling to calculate m-best assignment hypotheses/components
        uasses, nlcost = gibbs_jointpredupdt(-np.log(jointcostm), int(filter.H_upd))
        uasses = uasses + 1
        uasses[uasses <= ntracks] = -np.inf  # set not born/track deaths to -inf assignment
        uasses[(uasses > ntracks) & (uasses <= 2 * ntracks)] = 0  # set survived+missed to 0 assignment
        # set survived+detected to assignment of measurement index from 1:|Z|
        uasses[uasses > 2 * ntracks] = uasses[uasses > 2 * ntracks] - 2 * ntracks

        # component updates
        glmb_nextupdate_w = np.zeros(len(nlcost))
        self.assign_prob = np.zeros(m)  # adaptive birth weight for each measurement
        assign_meas = np.zeros((m, len(nlcost)), dtype=int)  # store indexes of measurement assigned to a track

        # generate corrresponding jointly predicted/updated hypotheses/components
        for hidx in range(0, len(nlcost)):
            update_hypcmp_tmp = uasses[hidx, :]
            # hypothesis/component weight
            # Vo Ba-Ngu "An efficient implementation of the generalized labeled multi-Bernoulli filter." eq (20)
            omega_z = -model.lambda_c + m * np.log(model.lambda_c * model.pdf_c) - nlcost[hidx]
            # Get measurement index from uasses (make sure minus 1 from [mindices+1])
            meas_idx = update_hypcmp_tmp[update_hypcmp_tmp > 0].astype(int) - 1
            assign_meas[meas_idx, hidx] = 1
            glmb_nextupdate_w[hidx] = omega_z

        glmb_nextupdate_w = np.exp(glmb_nextupdate_w - logsumexp(glmb_nextupdate_w))  # normalize weights

        # The following implementation is optimized for GLMB to LMB (glmb2lmb)
        # Refer "The Labeled Multi-Bernoulli Filter, 2014"
        death_tracks = []
        for (i, target) in enumerate(self.tt_lmb):
            notinf_uasses_idxs = np.nonzero(uasses[:, i] >= 0)
            workon_uasses = uasses[notinf_uasses_idxs, i]
            workon_weights = glmb_nextupdate_w[notinf_uasses_idxs]

            u, inv = np.unique(workon_uasses, return_inverse=True)
            if len(u) == 0:  # no measurement association (including misdetection)
                death_tracks.append(self.tt_lmb[i])
                continue
            sums = np.zeros(len(u), dtype=workon_weights.dtype)
            np.add.at(sums, inv, workon_weights)

            # select gating measurement indexes appear in ranked assignment (u: variable)
            # 0 for mis detection, 1->n for measurement index
            _, select_idxs, _ = np.intersect1d(np.insert(target.gatemeas + 1, 0, 0), u, return_indices=True)

            # select gaussian mixtures that are in 'u'
            l_range = np.repeat(target.gm_len, len(u))
            stop_idxs = (select_idxs + 1) * target.gm_len
            select_idxs = np.repeat(stop_idxs - l_range.cumsum(), l_range) + np.arange(l_range.sum())
            target.select_gms(select_idxs)
            target.finalize_glmb2lmb(sums)
        for target in death_tracks:
            self.tt_lmb.remove(target)
        # END

    def clean_lmb(self, filter):
        # prune tracks with low existence probabilities
        # extract vector of existence probabilities from LMB track table
        rvect = np.zeros(len(self.tt_lmb))
        for tabidx in range(len(self.tt_lmb)):
            rvect[tabidx] = self.tt_lmb[tabidx].r

        idxkeep = np.nonzero(rvect > filter.track_threshold)[0]
        tt_lmb_out = [self.tt_lmb[i] for i in idxkeep]

        # enforce cap on maximum number of tracks
        if len(tt_lmb_out) > filter.T_max:
            idxkeep = np.argsort(-rvect)
            tt_lmb_out = [self.tt_lmb[i] for i in idxkeep]

        # cleanup tracks
        for tabidx in range(len(tt_lmb_out)):
            w, m, P = gaus_prune(tt_lmb_out[tabidx].w, tt_lmb_out[tabidx].m, tt_lmb_out[tabidx].P,
                                 filter.elim_threshold)
            w, m, P = gaus_merge(w, m, P, filter.merge_threshold)
            w, m, P = gaus_cap(w, m, P, filter.L_max)
            tt_lmb_out[tabidx].w, tt_lmb_out[tabidx].m, tt_lmb_out[tabidx].P = w, m, P

        self.tt_lmb = tt_lmb_out

    # END clean_lmb

    def extract_estimates(self):
        # extract estimates via MAP cardinality and corresponding tracks
        rvect = np.zeros(len(self.tt_lmb))
        for tabidx in range(len(self.tt_lmb)):
            rvect[tabidx] = self.tt_lmb[tabidx].r
        rvect = np.minimum(rvect, 1. - 1e-9)
        rvect = np.maximum(rvect, 1e-9)

        cdn = np.prod(1 - rvect) * esf(rvect / (1 - rvect))
        mode = np.argmax(cdn)
        N = min(len(rvect), mode)
        idxcmp = np.argsort(-rvect)
        X, L = np.zeros((4, N)), np.zeros((2, N), dtype=int)
        for n in range(N):
            idxtrk = np.argmax(self.tt_lmb[idxcmp[n]].w)
            X[:, n] = self.tt_lmb[idxcmp[n]].m[:, idxtrk]
            L[:, n] = self.tt_lmb[idxcmp[n]].l
        return X, N, L

    def display_diaginfo(self, k, est, filter, T_predict, T_posterior, T_clean):
        rvect = np.zeros(len(self.tt_lmb))
        for tabidx in range(len(self.tt_lmb)):
            rvect[tabidx] = self.tt_lmb[tabidx].r
        rvect = np.minimum(rvect, 1. - 1e-9)
        rvect = np.maximum(rvect, 1e-9)
        cdn = np.prod(1 - rvect) * esf(rvect / (1 - rvect))
        eap = np.arange(len(cdn)) @ cdn
        var = np.square(np.arange(len(cdn))) @ cdn - (np.arange(len(cdn)) @ cdn) ** 2
        if filter.run_flag is not 'silence':
            print(' time= ', str(k),
                  ' #eap cdn=' "{:10.4f}".format(eap),
                  ' #var cdn=' "{:10.4f}".format(var),
                  ' #est card=' "{:10.4f}".format(est.N[k]),
                  ' #trax pred=' "{:10.4f}".format(T_predict),
                  ' #trax post=' "{:10.4f}".format(T_posterior),
                  ' #trax updt=', "{:10.4f}".format(T_clean))

    def run(self, model, filter, meas, est):
        for k in range(0, meas.K):
            # joint predict and update, results in GLMB, convert to LMB
            self.jointlmbpredictupdate(model, filter, meas, k)
            T_predict = len(self.tt_lmb)+model.T_birth
            T_posterior = len(self.tt_lmb)

            # pruning, truncation and track cleanup
            self.clean_lmb(filter)
            T_clean = len(self.tt_lmb)

            # state estimation
            X, N, L = self.extract_estimates()
            est.X[k] = X
            est.N[k] = N
            est.L[k] = L

            # display diagnostics
            self.display_diaginfo(k, est, filter, T_predict,T_posterior,T_clean)

    def plot_tracks(self, model, filter, truth, meas, est):
        fig, axs1, axs2 = plot_truth_meas(model, truth, meas)
        for k in range(0, meas.K):
            # joint predict and update, results in GLMB, convert to LMB
            self.jointlmbpredictupdate(model, filter, meas, k)
            T_predict = len(self.tt_lmb) + model.T_birth
            T_posterior = len(self.tt_lmb)

            # pruning, truncation and track cleanup
            self.clean_lmb(filter)
            T_clean = len(self.tt_lmb)

            X, N, L = self.extract_estimates()
            est.N[k] = N

            # display diagnostics
            self.display_diaginfo(k, est, filter, T_predict, T_posterior, T_clean)

            for i in range(N):
                loc, label = X[:, i], L[:, i]
                label_color = float(hash(str(label[0]) + '.' + str(label[1])) % 256) / 256
                axs1.scatter(k, loc[0], marker='.', s=80, color=label_color * np.ones(3))
                axs2.scatter(k, loc[2], marker='.', s=80, color=label_color * np.ones(3))

                fig.canvas.draw()
                image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                cv2.imshow("Tracking", image)
                cv2.waitKey(1)
    # END
