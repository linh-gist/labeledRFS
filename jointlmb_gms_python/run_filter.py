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

class target:
    # track table for GLMB (cell array of structs for individual tracks)
    def __init__(self):
        self.x_dim = 4
        self.w = 0
        self.m = np.empty((self.x_dim, 1))
        self.P = np.empty((self.x_dim, self.x_dim))
        self.l = np.empty(0, dtype=int)
        self.r = 0
        self.gatemeas = np.empty(0, dtype=int)

class LMB:
    def __init__(self):
        # initial prior
        self.tt_lmb_update = []
        # 1, track table for GLMB (cell array of structs for individual tracks)
        self.glmb_update_tt = []
        self.glmb_update_w = np.array([1])  # 2, vector of GLMB component/hypothesis weights
        self.glmb_update_I = [np.array([])]  # 3, cell of GLMB component/hypothesis labels (labels are indices/entries in track table)
        self.glmb_update_n = np.array([0])  # 4, vector of GLMB component/hypothesis cardinalities
        # 5, cardinality distribution of GLMB (vector of cardinality distribution probabilities)
        self.glmb_update_cdn = np.array([1])

        self.tt_birth=[]
        self.assign_prob = None
        self.prob_birth = 0.03

    def jointlmbpredictupdate(self, model, filter, meas, k):
        #  generate birth tracks
        if self.assign_prob is None:
            self.tt_birth = [target() for i in range(meas.Z[k].shape[1])]
            for idx in range(meas.Z[k].shape[1]):
                self.tt_birth[idx].r = self.prob_birth  # existence probability of this birth
                self.tt_birth[idx].w = 1  # weights of Gaussians for birth track
                self.tt_birth[idx].m = np.array([[meas.Z[k][0][idx]], [0], [meas.Z[k][1][idx]], [0]])
                self.tt_birth[idx].P = model.P_birth[0].reshape(model.x_dim, model.x_dim, 1)  # covs of Gaussians for birth track
                self.tt_birth[idx].l = np.array([k, idx])

        # generate surviving tracks
        tt_survive = [target() for i in range(0, len(self.tt_lmb_update))]  # initialize cell array
        for tabsidx in range(0, len(self.tt_lmb_update)):
            # predicted existence probability for surviving track
            tt_survive[tabsidx].r = model.P_S * self.tt_lmb_update[tabsidx].r
            mtemp_predict, Ptemp_predict = kalman_predict_multiple(model, self.tt_lmb_update[tabsidx].m,
                                                                   self.tt_lmb_update[tabsidx].P)
            tt_survive[tabsidx].m = mtemp_predict  # means of Gaussians for surviving track
            tt_survive[tabsidx].P = Ptemp_predict  # covs of Gaussians for surviving track
            tt_survive[tabsidx].w = self.tt_lmb_update[tabsidx].w  # weights of Gaussians for surviving track
            tt_survive[tabsidx].l = self.tt_lmb_update[tabsidx].l  # track label

        # create predicted tracks - concatenation of birth and survival
        tt_predict = self.tt_birth + tt_survive  # copy track table back to GLMB struct

        # gating by tracks
        if filter.gate_flag:
            for tabidx in range(0, len(tt_predict)):
                gate_idx = gate_meas_gms_idx(meas.Z[k], filter.gamma, model, tt_predict[tabidx].m,
                                  tt_predict[tabidx].P)
                tt_predict[tabidx].gatemeas = gate_idx.astype(int)
        else:
            for tabidx in range(0, len(tt_predict)):
                tt_predict[tabidx].gatemeas = range(0, meas.Z[k].shape[1])

        # precalculation loop for average survival/death probabilities
        avps = np.zeros((len(tt_predict), 1))
        for tabidx in range(0, len(tt_predict)):
            avps[tabidx] = tt_predict[tabidx].r
        avqs = 1 - avps

        # precalculation loop for average detection/missed probabilities
        avpd = np.zeros((len(tt_predict), 1));
        for tabidx in range(0, len(tt_predict)):
            avpd[tabidx] = model.P_D
        avqd = 1 - avpd

        # create updated tracks (single target Bayes update)
        m = meas.Z[k].shape[1]  # number of measurements
        tt_update = [[] for i in range((1 + m) * len(tt_predict))]  # initialize cell array
        # missed detection tracks (legacy tracks)
        for tabidx in range(0, len(tt_predict)):
            tt_update[tabidx] = deepcopy(tt_predict[tabidx])  # same track table
        # measurement updated tracks (all pairs)
        allcostm = np.zeros((len(tt_predict), m));
        for tabidx in range(0, len(tt_predict)):
            for emm in tt_predict[tabidx].gatemeas:
                # index of predicted track i updated with measurement j is (number_predicted_tracks*j + i)
                stoidx = len(tt_predict) * (emm + 1) + tabidx;
                # kalman update for this track and this measurement
                qz_temp, m_temp, P_temp = kalman_update_multiple(meas.Z[k][:, emm], model, tt_predict[tabidx].m,
                                                                 tt_predict[tabidx].P)
                # unnormalized updated weights
                w_temp = np.multiply(qz_temp, tt_predict[tabidx].w) + np.spacing(1)
                tt_update[stoidx] = target()
                tt_update[stoidx].m = m_temp  # means of Gaussians for updated track
                tt_update[stoidx].P = P_temp  # covs of Gaussians for updated track
                tt_update[stoidx].w = w_temp / sum(w_temp)  # weights of Gaussians for updated track
                tt_update[stoidx].l = tt_predict[tabidx].l  # track label
                allcostm[tabidx, emm] = sum(w_temp)  # predictive likelihood

        self.glmb_update_tt = tt_update  # copy track table back to GLMB struct
        # joint cost matrix
        jointcostm = np.concatenate((np.diagflat(avqs),
                               np.diagflat(np.multiply(avps, avqd)),
                               np.multiply(np.tile(np.multiply(avps, avpd), (1, m)), allcostm) / (
                                       model.lambda_c * model.pdf_c)), axis=1)
        # gated measurement index matrix
        gatemeasidxs = -1*np.ones((len(tt_predict), m), dtype=int)
        for tabidx in range(0, len(tt_predict)):
            gatemeasidxs[tabidx, 0:len(tt_predict[tabidx].gatemeas)] = tt_predict[tabidx].gatemeas
        gatemeasindc = gatemeasidxs >= 0

        # component updates
        glmb_nextupdate_w = np.empty(0)
        glmb_nextupdate_I = []
        glmb_nextupdate_n = np.empty(0, dtype=int)
        self.assign_prob = np.zeros(m)  # adaptive birth weight for each measurement

        # calculate best updated hypotheses/components
        cpreds = len(tt_predict)
        nbirths = len(self.tt_birth)
        nexists = len(self.tt_lmb_update)
        ntracks = nbirths + nexists
        tindices = np.concatenate((np.arange(0, nbirths), nbirths + np.arange(nexists))).astype(int)  # indices of all births and existing tracks  for current component
        lselmask = np.zeros((len(tt_predict), m), dtype=bool)
        lselmask[tindices, :] = gatemeasindc[tindices, :]  # logical selection mask to index gating matrices
        # union indices of gated measurements for corresponding tracks
        mindices = unique_faster(gatemeasidxs[lselmask])
        # cost matrix - [no_birth/is_death | born/survived+missed | born/survived+detected]
        take_rows = jointcostm[tindices]
        costm = take_rows[:, np.concatenate((tindices, cpreds + tindices, 2 * cpreds + mindices))]
        neglogcostm = -np.log(costm)  # negative log cost
        hypothesis_num = round(filter.H_upd)
        # murty's algo/gibbs sampling to calculate m-best assignment hypotheses/components
        uasses, nlcost = gibbs_jointpredupdt(neglogcostm, hypothesis_num)
        uasses = uasses + 1
        uasses[uasses <= ntracks] = -np.inf  # set not born/track deaths to -inf assignment
        uasses[(uasses > ntracks) & (uasses <= 2 * ntracks)] = 0  # set survived+missed to 0 assignment
        # set survived+detected to assignment of measurement index from 1:|Z|
        uasses[uasses > 2 * ntracks] = uasses[uasses > 2 * ntracks] - 2 * ntracks
        # restore original indices of gated measurements
        uasses[uasses > 0] = mindices[uasses[uasses > 0].astype(int) - 1] + 1

        # generate corrresponding jointly predicted/updated hypotheses/components
        for hidx in range(0, len(nlcost)):
            update_hypcmp_tmp = uasses[hidx, :]
            update_hypcmp_idx = cpreds * update_hypcmp_tmp + np.concatenate(
                (np.arange(0, nbirths), nbirths + np.arange(nexists))).astype(int)
            # hypothesis/component weight
            # Vo Ba-Ngu "An efficient implementation of the generalized labeled multi-Bernoulli filter." eq (20)
            omega_z = -model.lambda_c + m * np.log(model.lambda_c * model.pdf_c) - nlcost[hidx]
            # Get measurement index from uasses (make sure minus 1 from [mindices+1])
            meas_idx = update_hypcmp_tmp[update_hypcmp_tmp > 0].astype(int) - 1
            if len(meas_idx > 0):
                self.assign_prob[meas_idx] += np.exp(omega_z)
            glmb_nextupdate_w = np.append(glmb_nextupdate_w, omega_z)
            # hypothesis/component tracks (via indices to track table)
            glmb_nextupdate_I.append(update_hypcmp_idx[update_hypcmp_idx >= 0].astype(int))
            # hypothesis/component cardinality
            glmb_nextupdate_n = np.append(glmb_nextupdate_n, sum(update_hypcmp_idx >= 0))

        # create birth tracks
        self.apdative_birth(self.assign_prob, meas, model, k)

        glmb_nextupdate_w = np.exp(glmb_nextupdate_w - logsumexp(glmb_nextupdate_w))  # normalize weights

        # extract cardinality distribution
        glmb_nextupdate_cdn = np.zeros(max(glmb_nextupdate_n))
        for card in range(0, max(glmb_nextupdate_n)):
            # extract probability of n targets
            glmb_nextupdate_cdn[card] = sum(glmb_nextupdate_w[glmb_nextupdate_n == card])

        self.glmb_update_w = glmb_nextupdate_w  # 2
        self.glmb_update_I = glmb_nextupdate_I  # 3
        self.glmb_update_n = glmb_nextupdate_n  # 4
        self.glmb_update_cdn = glmb_nextupdate_cdn  # 5

        # remove duplicate entries and clean track table
        self.clean_predict()
        self.clean_update()

    def apdative_birth(self, assign_prob, meas, model, k):
        not_assigned_sum = sum(1 - assign_prob) + np.spacing(1)  # make sure this sum is not zero
        lambda_b = 0.1  # Set lambda_b to the mean cardinality of the birth multi-Bernoulli RFS
        self.tt_birth = [target() for i in range(assign_prob.shape[0])]
        for idx in range(meas.Z[k].shape[1]):
            # eq (75) "The Labeled Multi-Bernoulli Filter", Stephan Reuter∗, Ba-Tuong Vo, Ba-Ngu Vo, ...
            prob_birth = min(self.prob_birth, (1 - assign_prob[idx]) / not_assigned_sum * lambda_b)
            prob_birth = max(prob_birth, np.spacing(1))  # avoid zero birth probability
            self.tt_birth[idx].r = prob_birth  # existence probability of this birth
            self.tt_birth[idx].w = 1  # weights of Gaussians for birth track
            self.tt_birth[idx].m = np.array([[meas.Z[k][0][idx]], [0], [meas.Z[k][1][idx]], [0]])
            self.tt_birth[idx].P = model.P_birth[0].reshape(model.x_dim, model.x_dim, 1)  # covs of Gaussians for birth track
            self.tt_birth[idx].l = np.array([k, idx])
        # END

    def clean_predict(self):
        # hash label sets, find unique ones, merge all duplicates
        glmb_raw_hash = np.empty(0)
        for hidx in range(0, len(self.glmb_update_w)):
            hash_str = np.array2string(np.sort(self.glmb_update_I[hidx]), separator='*')[1:-1] + '*'
            glmb_raw_hash = np.append(glmb_raw_hash, hash_str)

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
        newindices = np.zeros(len(self.glmb_update_tt), dtype=int);
        newindices[usedindicator > 0] = np.arange(0, trackcount);
        # glmb_clean = np.zeros(len(se), dtype=self.dtype_tt);
        glmb_clean_tt = [t for t, useidx in zip(self.glmb_update_tt, usedindicator) if useidx>0]
        #self.glmb_update_tt[usedindicator > 0];
        self.glmb_update_tt = glmb_clean_tt  # 1

        glmb_clean_I = []
        for hidx in range(0, len(self.glmb_update_w)):
            glmb_clean_I.append(newindices[self.glmb_update_I[hidx]]);

        self.glmb_update_I = glmb_clean_I  # 3

    def glmb2lmb(self):
        # find unique labels (with different possibly different association histories)
        lmat = np.zeros((2,len(self.glmb_update_tt)), dtype=int)
        for tabidx in range(len(self.glmb_update_tt)):
            lmat[:,tabidx] = self.glmb_update_tt[tabidx].l

        cu, _, ic = np.unique(lmat, return_index=True, return_inverse=True, axis=1) # unique column

        # initialize LMB struct
        tt_lmb= []
        for i in range(cu.shape[1]):
            t = target()
            t.l = cu[:,i]
            tt_lmb.append(t)

        # extract individual tracks
        for hidx in range(len(self.glmb_update_w)):
           for t in range(self.glmb_update_n[hidx]):
              trkidx = self.glmb_update_I[hidx][t]
              newidx = ic[trkidx]
              tt_lmb[newidx].m = np.concatenate((tt_lmb[newidx].m,self.glmb_update_tt[trkidx].m), axis=1)
              tt_lmb[newidx].P = np.dstack((tt_lmb[newidx].P, self.glmb_update_tt[trkidx].P))
              tt_lmb[newidx].w = np.append(tt_lmb[newidx].w,
                                          self.glmb_update_w[hidx]*self.glmb_update_tt[trkidx].w)

        # extract existence probabilities and normalize track weights
        for tabidx in range(len(tt_lmb)):
           tt_lmb[tabidx].r= sum(tt_lmb[tabidx].w)
           tt_lmb[tabidx].w= tt_lmb[tabidx].w/tt_lmb[tabidx].r

        self.tt_lmb_update = tt_lmb

    def clean_lmb(self, filter):
        # prune tracks with low existence probabilities

        # extract vector of existence probabilities from LMB track table
        rvect = np.zeros(len(self.tt_lmb_update))
        for tabidx in range(len(self.tt_lmb_update)):
            rvect[tabidx] = self.tt_lmb_update[tabidx].r

        idxkeep= np.nonzero(rvect > filter.track_threshold)[0]
        tt_lmb_out= [self.tt_lmb_update[i] for i in idxkeep]

        # enforce cap on maximum number of tracks
        if len(tt_lmb_out) > filter.T_max:
            idxkeep = np.argsort(-rvect)
            tt_lmb_out= [self.tt_lmb_update[i] for i in idxkeep]

        # cleanup tracks
        for tabidx in range(len(tt_lmb_out)):
            w, m, P = gaus_prune(tt_lmb_out[tabidx].w, tt_lmb_out[tabidx].m, tt_lmb_out[tabidx].P, filter.elim_threshold)
            w, m, P = gaus_merge(w, m, P, filter.merge_threshold)
            w, m, P = gaus_cap(w, m, P, filter.L_max)
            tt_lmb_out[tabidx].w, tt_lmb_out[tabidx].m, tt_lmb_out[tabidx].P = w, m, P

        self.tt_lmb_update = tt_lmb_out
    # END clean_lmb

    def extract_estimates(self):
        # extract estimates via MAP cardinality and corresponding tracks
        rvect = np.zeros(len(self.tt_lmb_update))
        for tabidx in range(len(self.tt_lmb_update)):
            rvect[tabidx] = self.tt_lmb_update[tabidx].r
        rvect = np.minimum(rvect, 1. - 1e-9)
        rvect = np.maximum(rvect, 1e-9)

        cdn = np.prod(1 - rvect) * esf(rvect / (1 - rvect))
        mode = np.argmax(cdn)
        N = min(len(rvect), mode)
        idxcmp = np.argsort(-rvect)
        X, L = np.zeros((4, N)), np.zeros((2, N), dtype=int)
        for n in range(N):
            idxtrk = np.argmax(self.tt_lmb_update[idxcmp[n]].w)
            X[:, n] = self.tt_lmb_update[idxcmp[n]].m[:, idxtrk]
            L[:, n] = self.tt_lmb_update[idxcmp[n]].l
        return X, N, L

    def display_diaginfo(self, k, est, filter, T_predict, T_posterior, T_clean):
        rvect = np.zeros(len(self.tt_lmb_update))
        for tabidx in range(len(self.tt_lmb_update)):
            rvect[tabidx] = self.tt_lmb_update[tabidx].r
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
            T_predict = len(self.tt_lmb_update)+len(self.tt_birth)
            self.glmb2lmb()
            T_posterior = len(self.tt_lmb_update)

            # pruning, truncation and track cleanup
            self.clean_lmb(filter);
            T_clean = len(self.tt_lmb_update)

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
            T_predict = len(self.tt_lmb_update) + len(self.tt_birth)
            self.glmb2lmb()
            T_posterior = len(self.tt_lmb_update)

            # pruning, truncation and track cleanup
            self.clean_lmb(filter);
            T_clean = len(self.tt_lmb_update)

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
