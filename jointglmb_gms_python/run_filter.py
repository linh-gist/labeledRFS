from copy import deepcopy

import cv2
from scipy.stats.distributions import chi2
from gen_model import model
import numpy as np

from plot_results import plot_truth_meas
from utils import gate_meas_gms_idx, kalman_update_multiple, unique_faster, \
    gibbswrap_jointpredupdt_custom, intersect_mtlb, setxor_mtlb, kalman_update_single, murty, kalman_predict_multiple, \
    kalman_predict_single, new_dict_by_index
from scipy.special import logsumexp


class filter:
    # filter parameters
    def __init__(self, model):
        self.H_upd = 1000;  # requested number of updated components/hypotheses
        self.H_max = 1000;  # cap on number of posterior components/hypotheses
        self.hyp_threshold = 1e-15;  # pruning threshold for components/hypotheses

        self.L_max = 100;  # limit on number of Gaussians in each track - not implemented yet
        self.elim_threshold = 1e-5;  # pruning threshold for Gaussians in each track - not implemented yet
        self.merge_threshold = 4;  # merging threshold for Gaussians in each track - not implemented yet

        self.P_G = 0.9999999;  # gate size in percentage
        self.gamma = chi2.ppf(self.P_G, model.z_dim);  # inv chi^2 dn gamma value
        self.gate_flag = 1;  # gating on or off 1/0

        self.run_flag = 'disp';  # 'disp' or 'silence' for on the fly output


class est:
    def __init__(self, meas):
        self.X = [];
        self.N = np.zeros(meas.K);
        self.L = [];
        self.T = [];
        self.M = 0;
        self.J = np.empty((2,0), dtype=int);
        self.H = np.empty(0);


class target:
    # track table for GLMB (cell array of structs for individual tracks)
    def __init__(self):
        self.x_dim = 4
        self.w = 0
        self.m = np.empty((self.x_dim, 1))
        self.P = np.empty((self.x_dim, self.x_dim))
        self.l = np.empty(0)
        self.ah = np.empty(0)
        self.gatemeas = np.empty(0)

class GLMB:  # delta GLMB
    def __init__(self):
        # initial prior

        # 1, track table for GLMB (cell array of structs for individual tracks)
        self.glmb_update_tt = []
        self.glmb_update_w = np.array([1]);  # 2, vector of GLMB component/hypothesis weights
        self.glmb_update_I = [np.array([])];  # 3, cell of GLMB component/hypothesis labels (labels are indices/entries in track table)
        self.glmb_update_n = np.array([0]);  # 4, vector of GLMB component/hypothesis cardinalities
        # 5, cardinality distribution of GLMB (vector of cardinality distribution probabilities)
        self.glmb_update_cdn = np.array([1]);

    def jointpredictupdate(self, model, filter, meas, k):
        # ---generate next update
        # create birth tracks
        tt_birth = [target() for i in range(0, len(model.r_birth))]  # initialize cell array
        for tabbidx in range(0, len(model.r_birth)):
            tt_birth[tabbidx].m = model.m_birth[tabbidx];  # means of Gaussians for birth track
            tt_birth[tabbidx].P = model.P_birth[tabbidx];  # covs of Gaussians for birth track
            tt_birth[tabbidx].w = model.w_birth[tabbidx];  # weights of Gaussians for birth track
            tt_birth[tabbidx].l = np.array([k, tabbidx]);  # track label
            tt_birth[tabbidx].ah = np.array([]);  # track association history (empty at birth)

        # create surviving tracks - via time prediction (single target CK)
        tt_survive = [target() for i in range(0, len(self.glmb_update_tt))] # initialize cell array
        for tabsidx in range(0, len(self.glmb_update_tt)):
            # kalman prediction for GM
            mtemp_predict, Ptemp_predict = kalman_predict_single(model.F, model.Q, self.glmb_update_tt[tabsidx].m,
                                                                   self.glmb_update_tt[tabsidx].P);
            tt_survive[tabsidx].m = mtemp_predict;  # means of Gaussians for surviving track
            tt_survive[tabsidx].P = Ptemp_predict;  # covs of Gaussians for surviving track
            tt_survive[tabsidx].w = self.glmb_update_tt[tabsidx].w;  # weights of Gaussians for surviving track
            tt_survive[tabsidx].l = self.glmb_update_tt[tabsidx].l;  # track label
            # track association history (no change at prediction)
            tt_survive[tabsidx].ah = self.glmb_update_tt[tabsidx].ah;

        # create predicted tracks - concatenation of birth and survival
        glmb_predict_tt = tt_birth + tt_survive  # copy track table back to GLMB struct

        # gating by tracks
        if filter.gate_flag:
            for tabidx in range(0, len(glmb_predict_tt)):
                gate_idx = gate_meas_gms_idx(meas.Z[k], filter.gamma, model, glmb_predict_tt[tabidx].m,
                                  glmb_predict_tt[tabidx].P);
                glmb_predict_tt[tabidx].gatemeas = gate_idx
        else:
            for tabidx in range(0, len(glmb_predict_tt)):
                glmb_predict_tt[tabidx].gatemeas = range(0, meas.Z[k].shape[1]);

        # precalculation loop for average survival/death probabilities
        avps = np.concatenate((model.r_birth, np.zeros((len(self.glmb_update_tt), 1))));
        for tabidx in range(0, len(self.glmb_update_tt)):
            avps[model.T_birth + tabidx] = model.P_S;
        avqs = 1 - avps;

        # precalculation loop for average detection/missed probabilities
        avpd = np.zeros((len(glmb_predict_tt), 1));
        for tabidx in range(0, len(glmb_predict_tt)):
            avpd[tabidx] = model.P_D;
        avqd = 1 - avpd;

        # create updated tracks (single target Bayes update)
        m = meas.Z[k].shape[1];  # number of measurements
        tt_update = [[] for i in range((1 + m) * len(glmb_predict_tt))]  # initialize cell array
        # missed detection tracks (legacy tracks)
        for tabidx in range(0, len(glmb_predict_tt)):
            tt_update[tabidx] = deepcopy(glmb_predict_tt[tabidx]);  # same track table
            # track association history (updated for missed detection)
            tt_update[tabidx].ah = np.append(tt_update[tabidx].ah, -1).astype(int);
        # measurement updated tracks (all pairs)
        allcostm = np.zeros((len(glmb_predict_tt), m));
        for tabidx in range(0, len(glmb_predict_tt)):
            for emm in glmb_predict_tt[tabidx].gatemeas:
                # index of predicted track i updated with measurement j is (number_predicted_tracks*j + i)
                stoidx = len(glmb_predict_tt) * (emm + 1) + tabidx;
                # kalman update for this track and this measurement
                qz_temp, m_temp, P_temp = kalman_update_single(meas.Z[k][:, emm], model.H, model.R, glmb_predict_tt[tabidx].m,
                                                                 glmb_predict_tt[tabidx].P);
                # unnormalized updated weights
                w_temp = np.multiply(qz_temp, glmb_predict_tt[tabidx].w) + np.spacing(1)
                tt_update[stoidx] = target()
                tt_update[stoidx].m = m_temp;  # means of Gaussians for updated track
                tt_update[stoidx].P = P_temp;  # covs of Gaussians for updated track
                tt_update[stoidx].w = w_temp / sum(w_temp);  # weights of Gaussians for updated track
                tt_update[stoidx].l = glmb_predict_tt[tabidx].l;  # track label
                tt_update[stoidx].ah = np.append(glmb_predict_tt[tabidx].ah, emm).astype(int);  # track association history (updated with new measurement)
                allcostm[tabidx, emm] = sum(w_temp);  # predictive likelihood

        glmb_nextupdate_tt = tt_update;  # copy track table back to GLMB struct
        # joint cost matrix
        jointcostm = np.concatenate((np.diagflat(avqs),
                               np.diagflat(np.multiply(avps, avqd)),
                               np.multiply(np.tile(np.multiply(avps, avpd), (1, m)), allcostm) / (
                                       model.lambda_c * model.pdf_c)), axis=1);
        # gated measurement index matrix
        gatemeasidxs = -1*np.ones((len(glmb_predict_tt), m), dtype=int);
        for tabidx in range(0, len(glmb_predict_tt)):
            gatemeasidxs[tabidx, 0:len(glmb_predict_tt[tabidx].gatemeas)] = glmb_predict_tt[tabidx].gatemeas;
        gatemeasindc = gatemeasidxs >= 0;

        # component updates
        runidx = 0;
        glmb_nextupdate_w = np.empty(0)
        glmb_nextupdate_I = []
        glmb_nextupdate_n = np.empty(0, dtype=int)
        for pidx in range(0, len(self.glmb_update_w)):
            # calculate best updated hypotheses/components
            cpreds = len(glmb_predict_tt);
            nbirths = model.T_birth;
            nexists = len(self.glmb_update_I[pidx]);
            ntracks = nbirths + nexists;
            tindices = np.concatenate((np.arange(0, nbirths), nbirths + self.glmb_update_I[
                pidx])).astype(int)  # indices of all births and existing tracks  for current component
            lselmask = np.zeros((len(glmb_predict_tt), m), dtype=bool);
            lselmask[tindices, :] = gatemeasindc[tindices, :];  # logical selection mask to index gating matrices
            # union indices of gated measurements for corresponding tracks
            mindices = unique_faster(gatemeasidxs[lselmask]);
            # cost matrix - [no_birth/is_death | born/survived+missed | born/survived+detected]
            take_rows = jointcostm[tindices]
            costm = take_rows[:, np.concatenate((tindices, cpreds + tindices, 2 * cpreds + mindices))];
            neglogcostm = -np.log(costm);  # negative log cost
            hypothesis_num = round(filter.H_upd * np.sqrt(self.glmb_update_w[pidx]) / sum(np.sqrt(self.glmb_update_w)))
            # murty's algo/gibbs sampling to calculate m-best assignment hypotheses/components
            uasses, nlcost = murty(neglogcostm, hypothesis_num);
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
                    (np.arange(0, nbirths), nbirths + self.glmb_update_I[pidx])).astype(int)
                # hypothesis/component weight
                cmpt = -model.lambda_c + m * np.log(model.lambda_c * model.pdf_c) + np.log(self.glmb_update_w[pidx]) - \
                       nlcost[hidx]
                glmb_nextupdate_w = np.append(glmb_nextupdate_w, cmpt)
                # hypothesis/component tracks (via indices to track table)
                glmb_nextupdate_I.append(update_hypcmp_idx[update_hypcmp_idx >= 0].astype(int));
                # hypothesis/component cardinality
                glmb_nextupdate_n = np.append(glmb_nextupdate_n, sum(update_hypcmp_idx >= 0));
                runidx = runidx + 1;

        glmb_nextupdate_w = np.exp(glmb_nextupdate_w - logsumexp(glmb_nextupdate_w));  # normalize weights

        # extract cardinality distribution
        glmb_nextupdate_cdn = np.zeros(max(glmb_nextupdate_n))
        for card in range(0, max(glmb_nextupdate_n)):
            glmb_nextupdate_cdn[card] = sum(
                glmb_nextupdate_w[glmb_nextupdate_n == card]);  # extract probability of n targets

        # copy glmb update to the next time step
        self.glmb_update_tt = glmb_nextupdate_tt  # 1
        self.glmb_update_w = glmb_nextupdate_w  # 2
        self.glmb_update_I = glmb_nextupdate_I  # 3
        self.glmb_update_n = glmb_nextupdate_n  # 4
        self.glmb_update_cdn = glmb_nextupdate_cdn  # 5

        # remove duplicate entries and clean track table
        self.clean_predict()
        self.clean_update()

    def clean_predict(self):
        # hash label sets, find unique ones, merge all duplicates
        glmb_raw_hash = np.empty(0)
        for hidx in range(0, len(self.glmb_update_w)):
            hash_str = np.array2string(np.sort(self.glmb_update_I[hidx]), separator='*')[1:-1] + '*'
            glmb_raw_hash = np.append(glmb_raw_hash, hash_str);

        cu, _, ic = np.unique(glmb_raw_hash, return_index=True, return_inverse=True, axis=0)

        glmb_temp_w = np.zeros((len(cu)));
        glmb_temp_I = [np.array([]) for i in range(0, len(ic))];
        glmb_temp_n = np.zeros((len(cu)), dtype=int);
        for hidx in range(0, len(ic)):
            glmb_temp_w[ic[hidx]] = glmb_temp_w[ic[hidx]] + self.glmb_update_w[hidx];
            glmb_temp_I[ic[hidx]] = self.glmb_update_I[hidx];
            glmb_temp_n[ic[hidx]] = self.glmb_update_n[hidx];

        self.glmb_update_w = glmb_temp_w  # 2
        self.glmb_update_I = glmb_temp_I  # 3
        self.glmb_update_n = glmb_temp_n  # 4

    def clean_update(self):
        # flag used tracks
        usedindicator = np.zeros(len(self.glmb_update_tt), dtype=int);
        for hidx in range(0, len(self.glmb_update_w)):
            usedindicator[self.glmb_update_I[hidx]] = usedindicator[self.glmb_update_I[hidx]] + 1;
        trackcount = sum(usedindicator > 0);

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

    def prune(self, filter):
        # prune components with weights lower than specified threshold
        idxkeep = np.nonzero(self.glmb_update_w > filter.hyp_threshold)[0];
        glmb_out_w = self.glmb_update_w[idxkeep];
        glmb_out_I = [self.glmb_update_I[i] for i in idxkeep]
        glmb_out_n = self.glmb_update_n[idxkeep];

        glmb_out_w = glmb_out_w / sum(glmb_out_w);
        glmb_out_cdn = np.zeros((max(glmb_out_n)))
        for card in range(0, np.max(glmb_out_n)):
            glmb_out_cdn[card] = sum(glmb_out_w[glmb_out_n == card]);

        self.glmb_update_w = glmb_out_w  # 2
        self.glmb_update_I = glmb_out_I  # 3
        self.glmb_update_n = glmb_out_n  # 4
        self.glmb_update_cdn = glmb_out_cdn  # 5

    def cap(self, filter):
        # cap total number of components to specified maximum
        if len(self.glmb_update_w) > filter.H_max:
            idxsort = np.argsort(-self.glmb_update_w);
            idxkeep = idxsort[0:filter.H_max];
            # glmb_out.tt= glmb_in.tt;
            glmb_out_w = self.glmb_update_w[idxkeep];
            glmb_out_I = self.glmb_update_I[idxkeep];
            glmb_out_n = self.glmb_update_n[idxkeep];

            glmb_out_w = glmb_out_w / sum(glmb_out_w);
            glmb_out_cdn = np.zeros(max(glmb_out_n))
            for card in range(0, max(glmb_out_n)):
                glmb_out_cdn[card] = sum(glmb_out_w[glmb_out_n == card]);

            self.glmb_update_w = glmb_out_w  # 2
            self.glmb_update_I = glmb_out_I  # 3
            self.glmb_update_n = glmb_out_n  # 4
            self.glmb_update_cdn = glmb_out_cdn  # 5

    def extract_estimates_recursive(self, model, meas, est):
        # extract estimates via recursive estimator, where
        # trajectories are extracted via association history, and
        # track continuity is guaranteed with a non-trivial estimator

        # extract MAP cardinality and corresponding highest weighted component
        mode = np.argmax(self.glmb_update_cdn);
        M = mode;
        T = [np.array([], dtype=int) for i in range(0, M)];
        J = np.zeros((2, M), dtype=int);

        idxcmp = np.argmax(np.multiply(self.glmb_update_w, (self.glmb_update_n == M).astype(int)));
        for m in range(0, M):
            idxptr = self.glmb_update_I[idxcmp][m];
            T[m] = self.glmb_update_tt[idxptr].ah;
            J[:, m] = self.glmb_update_tt[idxptr].l;

        H = np.empty(0);
        for m in range(0, M):
            H = np.append(H, (str(J[0, m]) + '.' + str(J[1, m])));

        # compute dead & updated & new tracks
        _, iio, iis = np.intersect1d(est.H, H, assume_unique=False, return_indices=True)
        _, iid, iin = setxor_mtlb(est.H, H)

        est.M = M;
        est.T = [est.T[i] for i in iid]
        est.T += [T[i] for i in iis]
        est.T += [T[i] for i in iin]
        est.J = np.column_stack((est.J[:, iid], J[:, iis], J[:, iin]));
        est.H = np.concatenate((est.H[iid], H[iis], H[iin]));

        # write out estimates in standard format
        est.N = np.zeros(meas.K);
        est.X = {key: np.empty((model.x_dim, 0)) for key in range(0, meas.K)};
        est.L = {key: np.empty((model.z_dim, 0), dtype=int) for key in range(0, meas.K)};
        for t in range(0, len(est.T)):
            ks = est.J[0, t];
            bidx = est.J[1, t];
            tah = est.T[t];

            w = model.w_birth[bidx];
            m = model.m_birth[bidx];
            P = model.P_birth[bidx];
            for u in range(0, len(tah)):
                m, P = kalman_predict_single(model.F, model.Q, m, P);
                k = ks + u;
                emm = tah[u];
                if emm >= 0:
                    [qz, m, P] = kalman_update_single(meas.Z[k][:, emm], model.H, model.R, m, P);
                    w = np.multiply(qz, w + np.spacing(1));
                    w = w / sum(w);

                idxtrk = np.argmax(w);
                est.N[k] = est.N[k] + 1;
                est.X[k] = np.column_stack((est.X[k], m[:, idxtrk]));
                est.L[k] = np.column_stack((est.L[k], est.J[:, t]));
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
                  ' #trax updt=', "{:10.4f}".format(len(self.glmb_update_tt)));

    def run(self, model, filter, meas, est):
        for k in range(0, meas.K):
            # joint prediction and update
            self.jointpredictupdate(model, filter, meas, k);
            H_posterior = len(self.glmb_update_w);

            # pruning and truncation
            self.prune(filter);
            H_prune = len(self.glmb_update_w);
            self.cap(filter);
            H_cap = len(self.glmb_update_w);

            # state estimation and display diagnostics
            # [est.X{k},est.N(k),est.L{k}]= extract_estimates(glmb_update,model);
            est = self.extract_estimates_recursive(model, meas, est);
            self.display_diaginfo(k, est, filter, H_posterior, H_posterior, H_prune, H_cap);

    def plot_tracks(self, model, filter, truth, meas):
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
