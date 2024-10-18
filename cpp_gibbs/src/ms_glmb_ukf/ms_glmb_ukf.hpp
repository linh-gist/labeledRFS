//
// Created by linh on 2022-04-03.
//

#ifndef UKF_TARGET_MS_GLMB_UKF_HPP
#define UKF_TARGET_MS_GLMB_UKF_HPP

#include <set>
#include <unordered_map>
#include <numeric> // iota
#include "Filter.hpp"
#include "Model.hpp"
#include "Target.hpp"
//#include "gibbs_multisensor.hpp"

VectorXi ndsub2ind(VectorXi siz, MatrixXi idx) {
    if (idx.cols() == 0) {
        return VectorXi(0);
    } else {
        VectorXi linidx = idx.col(0);
        int cumprod = siz(0);
        for (int i = 1; i < idx.cols(); i++) {
            linidx = linidx + idx.col(i) * cumprod;
            cumprod = (cumprod * siz(i));
        }
        return linidx;
    }
}

double log_sum_exp(VectorXd arr) {
    int count = arr.size();
    if (count > 0) {
        double maxVal = arr.maxCoeff();
        double sum = 0;
        for (int i = 0; i < count; i++) {
            sum += exp(arr(i) - maxVal);
        }
        return log(sum) + maxVal;
    } else {
        return 0.0;
    }
}

class MSGLMB {
private:
    vector<Target> glmb_update_tt;   // (1) track table for GLMB (individual tracks)
    VectorXd glmb_update_w;          // (2) vector of GLMB component/hypothesis weights
    vector<VectorXi> glmb_update_I;  // (3) cell of GLMB component/hypothesis labels (in track table)
    VectorXi glmb_update_n;          // (4) vector of GLMB component/hypothesis cardinalities
    VectorXd glmb_update_cdn;        // (5) cardinality distribution of GLMB
    vector<Target> tt_birth;
    Model model;
    Filter filter;

    void msjointpredictupdate(Model model, Filter filter, vector<MatrixXd> measZ, int k) {
        // create birth tracks
        tt_birth.resize(0);
        for (int tabbidx = 0; tabbidx < model.m.size(); tabbidx++) {
            Vector2i label;
            label << k, tabbidx;
            Target tt(model.m[tabbidx], model.P, 0.0001, 0.99, 0.98, label, model);
            tt_birth.push_back(tt);
        }
        // create surviving tracks - via time prediction (single target CK)
        for (Target &tt: glmb_update_tt) {
            tt.predict();
        }
        // create predicted tracks - concatenation of birth and survival
        vector<Target> glmb_predict_tt(tt_birth);  // copy track table back to GLMB struct
        glmb_predict_tt.insert(glmb_predict_tt.end(), glmb_update_tt.begin(), glmb_update_tt.end());
        // precalculation loop for average survival/death probabilities
        int cpreds = glmb_predict_tt.size();
        VectorXd avps(cpreds);
        // precalculation loop for average detection/missed probabilities
        MatrixXd avpd = MatrixXd::Zero(cpreds, model.N_sensors);
        for (int tabidx = 0; tabidx < tt_birth.size(); tabidx++) {
            avps(tabidx) = glmb_predict_tt[tabidx].r;
            for (int s = 0; s < model.N_sensors; s++) {
                avpd(tabidx, s) = glmb_predict_tt[tabidx].PD;
            }
        }
        for (int tabidx = tt_birth.size(); tabidx < cpreds; tabidx++) {
            avps(tabidx) = glmb_predict_tt[tabidx].PS;
            for (int s = 0; s < model.N_sensors; s++) {
                avpd(tabidx, s) = glmb_predict_tt[tabidx].PD;
            }
        }
        VectorXd avqs = 1 - avps.array();
        MatrixXd avqd = 1 - avpd.array();

        // create updated tracks (single target Bayes update)
        // nested for loop over all predicted tracks and sensors - slow way
        // Kalman updates on the same prior recalculate all quantities
        VectorXi m(model.N_sensors);
        vector<MatrixXd> allcostc(model.N_sensors);
        // extra state for not detected for each sensor (meas "0" in pos 1)
        vector<MatrixXd> jointcostc(model.N_sensors);
        // posterior probability of target survival (after measurement updates)
        MatrixXd avpp = MatrixXd::Zero(cpreds, model.N_sensors);
        // gated measurement index matrix
        vector<MatrixXi> gatemeasidxs(model.N_sensors);
        for (int s = 0; s < model.N_sensors; s++) {
            m[s] = measZ[s].cols();  // number of measurements
            allcostc[s] = MatrixXd::Zero(cpreds, 1 + m(s));
            jointcostc[s] = MatrixXd::Zero(cpreds, 1 + m(s));

            gatemeasidxs[s] = MatrixXi::Ones(cpreds, m(s));
            gatemeasidxs[s] *= -1;

            for (int tabidx = 0; tabidx < cpreds; tabidx++) {
                Target &tt = glmb_predict_tt[tabidx];
                tt.gate_msmeas_ukf(measZ[s], s, filter.gamma, filter.ukf_alpha, filter.ukf_kappa, filter.ukf_beta);
                MatrixXd allcostc_temp = MatrixXd::Zero(cpreds, 1 + m(s));
                for (int emm = 0; emm < m(s); emm++) {
                    if (tt.gatemeas[s](emm) >= 0) {
                        // unnormalized updated weights
                        double w_temp = tt.ukf_update_per_sensor(measZ[s].col(emm), s, filter.ukf_alpha,
                                                                 filter.ukf_kappa, filter.ukf_beta);
                        // predictive likelihood
                        allcostc[s](tabidx, 1 + emm) = w_temp + std::nexttoward(0.0, 1.0L);
                    }
                }
                VectorXd cij_m = avps.array() * avqd.col(s).array(); // survived and missed detection
                allcostc_temp.col(0) = cij_m;
                VectorXd cij_d = avps.array() * avpd.col(s).array();
                allcostc_temp(all, seq(1, m(s))).colwise() += cij_d;  // survived and detected targets
                jointcostc[s] = allcostc_temp.array() * allcostc[s].array() / (model.lambda_c * model.pdf_c);
                jointcostc[s].col(0) = cij_m;
                avpp.col(s) = jointcostc[s].rowwise().sum();

                gatemeasidxs[s](tabidx, all) = tt.gatemeas[s];
            }
        }

        // component updates
        int runidx = 0;
        VectorXd glmb_nextupdate_w(filter.H_max * 2);
        vector<VectorXi> glmb_nextupdate_I;
        VectorXi glmb_nextupdate_n = VectorXi::Zero(filter.H_upd * 2);
        int nbirths = tt_birth.size();
        VectorXd sqrt_hypoth_num = glmb_update_w.array().sqrt();
        VectorXi hypoth_num = (filter.H_upd * sqrt_hypoth_num / sqrt_hypoth_num.sum()).cast<int>();

        vector<int> tt_update_parent;
        MatrixXi tt_update_currah(0, model.N_sensors);
        vector<int> tt_update_linidx;
        for (int pidx = 0; pidx < glmb_update_w.size(); pidx++) {
            // calculate best updated hypotheses/components
            int nexists = glmb_update_I[pidx].size();
            int ntracks = nbirths + nexists;
            // indices of all births and existing tracks  for current component
            VectorXi tindices(ntracks);
            tindices << VectorXi::LinSpaced(nbirths, 0, nbirths - 1), glmb_update_I[pidx].array() + nbirths;
            vector<VectorXi> mindices(model.N_sensors); // mindices, sorted -1, 0, 1, 2
            vector<MatrixXd> costc(model.N_sensors);
            for (int s = 0; s < model.N_sensors; s++) {
                // union indices of gated measurements for corresponding tracks
                MatrixXi gate_tindices = gatemeasidxs[s](tindices, all);
                std::set<int> mindices_set{gate_tindices.data(), gate_tindices.data() + gate_tindices.size()};
                std::vector<int> temp1(mindices_set.begin(), mindices_set.end()); // convert set to vector std
                VectorXi temp2 = VectorXi::Map(temp1.data(), temp1.size()); // convert vector std to eigen
                if (temp2(0) == -1) {
                    mindices[s] = VectorXi::Zero(temp2.size());
                    mindices[s](seq(1, temp2.size() - 1)) = 1 + temp2(seq(1, temp2.size() - 1)).array();
                } else {
                    mindices[s] = VectorXi::Zero(temp2.size() + 1);
                    mindices[s](seq(1, temp2.size())) = 1 + temp2.array();
                }

                costc[s] = jointcostc[s](tindices, mindices[s]);
            }
            VectorXd dcost = avqs(tindices); // death cost
            VectorXd scost = avpp(tindices, all).rowwise().prod(); // posterior survival cost
            VectorXd dprob = dcost.array() / (dcost + scost).array();
            vector<MatrixXi> uasses = gibbs_multisensor_approx_cheap(dprob, costc, hypoth_num[pidx]);

            MatrixXd local_avpdm = avpd(tindices, all);
            MatrixXd local_avqdm = avqd(tindices, all);
            MatrixXi aug_idx(tindices.size(), model.N_sensors + 1);
            aug_idx.col(0) = tindices;
            VectorXi sizeIdx(m.size() + 1);
            sizeIdx(0) = cpreds;
            sizeIdx(seq(1, m.size())) = m.array() + 1;
            for (int hidx = 0; hidx < uasses.size(); hidx++) {
                MatrixXi update_hypcmp_tmp = uasses[hidx]; // ntracks x N_sensors
                vector<int> off_vec, not_off_vec;
                double lambda_pdf_c = 0;
                for (int ivec = 0; ivec < update_hypcmp_tmp.rows(); ivec++) {
                    if (update_hypcmp_tmp(ivec, 0) < 0) { // check death target in only one sensor
                        off_vec.push_back(ivec);
                    } else if (update_hypcmp_tmp(ivec, 0) == 0) {
                        not_off_vec.push_back(ivec);
                    } else { // >0
                        not_off_vec.push_back(ivec);
                        lambda_pdf_c += log(model.lambda_c * model.pdf_c);
                    }
                }
                aug_idx(all, seq(1, model.N_sensors)) = update_hypcmp_tmp;
                VectorXi off_idx = VectorXi::Map(off_vec.data(), off_vec.size());
                VectorXi not_offidx = VectorXi::Map(not_off_vec.data(), not_off_vec.size());
                VectorXi update_hypcmp_idx = VectorXi::Zero(update_hypcmp_tmp.rows());
                update_hypcmp_idx(off_idx).array() = -1;
                update_hypcmp_idx(not_offidx) = ndsub2ind(sizeIdx, aug_idx(not_offidx, all));
                int num_trk = (update_hypcmp_idx.array() >= 0).cast<int>().sum();
                //
                double stemp = log(model.lambda_c * model.pdf_c) * m.array().sum();
                stemp += avps(tindices(not_offidx)).array().log().sum() + avqs(tindices(off_idx)).array().log().sum();
                stemp += (update_hypcmp_tmp.array() > 0).select(local_avpdm, 1).array().log().sum();
                stemp += (update_hypcmp_tmp.array() == 0).select(local_avqdm, 1).array().log().sum();
                stemp -= (lambda_pdf_c * model.N_sensors);
                glmb_nextupdate_w(runidx) = stemp + log(glmb_update_w(pidx)); // hypothesis/component weight
                if (num_trk > 0) {
                    // hypothesis/component tracks (via indices to track table)
                    int parent_size = tt_update_parent.size();
                    VectorXi range_temp = VectorXi::LinSpaced(num_trk, parent_size, parent_size + num_trk - 1);
                    glmb_nextupdate_I.push_back(range_temp);
                } else {
                    glmb_nextupdate_I.push_back(VectorXi(0));
                }
                glmb_nextupdate_n[runidx] = num_trk; // hypothesis / component cardinality
                for (int tt_idx = 0; tt_idx < not_offidx.size(); tt_idx++) {
                    tt_update_parent.push_back(tindices(not_offidx(tt_idx)));
                    tt_update_linidx.push_back(update_hypcmp_idx(not_offidx(tt_idx)));
                }
                int currah = tt_update_currah.rows(); // Resizes the matrix, while leaving old values untouched.
                tt_update_currah.conservativeResize(currah + not_offidx.size(), NoChange);
                tt_update_currah(seq(currah, currah + not_offidx.size() - 1), all) = update_hypcmp_tmp(not_offidx, all);
                runidx = runidx + 1;
            }
        }

        // component updates via posterior weight correction (including generation of track table)
        unordered_map<int, int> umap; // sorted, order
        int unique_idx = 0;
        vector<double> tt_update_msqz;
        vector<Target> tt_update;
        VectorXi ttU_newidx(tt_update_linidx.size());
        for (int tt_idx = 0; tt_idx < tt_update_linidx.size(); tt_idx++) {
            if (umap.find(tt_update_linidx[tt_idx]) == umap.end()) {
                umap[tt_update_linidx[tt_idx]] = unique_idx;

                int preidx = tt_update_parent[tt_idx];
                VectorXi meascomb = tt_update_currah.row(tt_idx);
                // kalman update for this track and all joint measurements
                double qz_temp;
                Target tt;
                glmb_predict_tt[preidx].ukf_msjointupdate(measZ, meascomb, filter.ukf_alpha, filter.ukf_kappa,
                                                          filter.ukf_beta, tt, qz_temp);
                tt_update_msqz.push_back(qz_temp);
                tt_update.push_back(tt);
                ttU_newidx(tt_idx) = unique_idx;

                unique_idx += 1;
            } else {
                ttU_newidx(tt_idx) = umap[tt_update_linidx[tt_idx]];
            }
        }
        VectorXd msqz = VectorXd::Map(tt_update_msqz.data(), tt_update_msqz.size());
        // normalize weights
        VectorXd glmb_w_runidx = glmb_nextupdate_w(seq(0, runidx - 1));
        for (int pidx = 0; pidx < runidx; pidx++) {
            glmb_nextupdate_I[pidx] = ttU_newidx(glmb_nextupdate_I[pidx]);
            glmb_w_runidx[pidx] = glmb_w_runidx[pidx] + msqz(glmb_nextupdate_I[pidx]).array().log().sum();
        }
        glmb_update_w = (glmb_w_runidx.array() - log_sum_exp(glmb_w_runidx)).exp(); // 2

        // extract cardinality distribution
        glmb_update_n = glmb_nextupdate_n(seq(0, runidx - 1)); // 4
        VectorXd glmb_nextupdate_cdn(glmb_update_n.maxCoeff() + 1);
        for (int card = 0; card < glmb_nextupdate_cdn.size(); card++) {
            // extract probability of n targets
            glmb_nextupdate_cdn[card] = (glmb_update_n.array() == card).select(glmb_update_w, 0).sum();
        }
        // copy glmb update to the next time step
        glmb_update_tt = tt_update;             // 1
        glmb_update_I = glmb_nextupdate_I;      // 3
        glmb_update_cdn = glmb_nextupdate_cdn;  // 5
        // remove duplicate entries and clean track table
        clean_predict();
        clean_update();
    }

    void clean_predict() {
        // hash label sets, find unique ones, merge all duplicates
        unordered_map<string, int> umap;
        VectorXd glmb_temp_w = VectorXd::Zero(glmb_update_w.size());
        vector<VectorXi> glmb_temp_I(glmb_update_I);
        VectorXi glmb_temp_n(glmb_update_n.size());
        int unique_idx = 0;
        string hash;
        hash.reserve(2048); // for a single (I), can have 500 objects with identity going up to 999
        for (int hidx = 0; hidx < glmb_update_w.size(); hidx++) {
            VectorXi glmb_I = glmb_update_I[hidx];
            std::sort(glmb_I.data(), glmb_I.data() + glmb_I.size());
            hash = "";
            for (int i = 0; i < glmb_I.size(); i++) {
                hash.append(to_string(glmb_I(i)));
                hash.append("*");
            }
            // If not present, then put it in unordered_set
            if (umap.find(hash) == umap.end()) {
                umap[hash] = unique_idx;
                glmb_temp_w[unique_idx] = glmb_update_w[hidx];
                glmb_temp_I[unique_idx] = glmb_update_I[hidx];
                glmb_temp_n[unique_idx] = glmb_update_n[hidx];
                unique_idx += 1;
            } else {
                glmb_temp_w[umap[hash]] += glmb_update_w[hidx];
            }
        }
        glmb_update_w = glmb_temp_w(seq(0, unique_idx - 1));  // 2
        glmb_temp_I.erase(glmb_temp_I.begin() + unique_idx, glmb_temp_I.end());
        glmb_update_I = glmb_temp_I;  // 3
        glmb_update_n = glmb_temp_n(seq(0, unique_idx - 1));  // 4
    }

    void clean_update() {
        // flag used tracks
        VectorXi usedindicator = VectorXi::Zero(glmb_update_tt.size());
        for (int hidx = 0; hidx < glmb_update_w.size(); hidx++) {
            usedindicator(glmb_update_I[hidx]).array() += 1;
        }
        // remove unused tracks and reindex existing hypotheses/components
        VectorXi newindices = VectorXi::Zero(glmb_update_tt.size());
        int new_idx = 0;
        vector<Target> glmb_clean_tt;
        for (int i = 0; i < newindices.size(); i++) {
            if (usedindicator(i) > 0) {
                newindices(i) = new_idx;
                new_idx += 1;
                glmb_clean_tt.push_back(glmb_update_tt[i]);
            }
        }
        for (int hidx = 0; hidx < glmb_update_w.size(); hidx++) {
            glmb_update_I[hidx] = newindices(glmb_update_I[hidx]);
        }
        glmb_update_tt = glmb_clean_tt;
    }

    void prune(Filter filter) {
        // prune components with weights lower than specified threshold
        vector<int> idxkeep;
        vector<VectorXi> glmb_out_I;
        for (int i = 0; i < glmb_update_I.size(); i++) {
            if (glmb_update_w(i) > filter.hyp_threshold) {
                idxkeep.push_back(i);
                glmb_out_I.push_back(glmb_update_I[i]);
            }
        }
        VectorXi idxkeep_eigen = VectorXi::Map(idxkeep.data(), idxkeep.size());
        VectorXd glmb_out_w = glmb_update_w(idxkeep_eigen);
        glmb_out_w = glmb_out_w / glmb_out_w.sum();
        VectorXi glmb_out_n = glmb_update_n(idxkeep_eigen);
        VectorXd glmb_out_cdn(glmb_out_n.maxCoeff() + 1);
        for (int card = 0; card < glmb_out_cdn.size(); card++) {
            glmb_out_cdn[card] = (glmb_out_n.array() == card).select(glmb_out_w, 0).sum();
        }
        glmb_update_w = glmb_out_w;  // 2
        glmb_update_I = glmb_out_I;  // 3
        glmb_update_n = glmb_out_n;  // 4
        glmb_update_cdn = glmb_out_cdn;  // 5
    }

    void cap(Filter filter) {
        // cap total number of components to specified maximum
        if (glmb_update_w.size() > filter.H_max) {
            // initialize original index locations
            vector<double> v_glmb_w(glmb_update_w.size());
            VectorXd::Map(&v_glmb_w[0], glmb_update_w.size()) = glmb_update_w;
            vector<int> idx(glmb_update_w.size());
            std::iota(idx.begin(), idx.end(), 0);
            stable_sort(idx.begin(), idx.end(),
                        [&v_glmb_w](int i1, int i2) { return v_glmb_w[i1] > v_glmb_w[i2]; });
            VectorXi idx_eigen = VectorXi::Map(idx.data(), idx.size());
            VectorXi idxkeep_eigen = idx_eigen(seq(0, filter.H_max - 1));

            VectorXd glmb_out_w = glmb_update_w(idxkeep_eigen);
            VectorXi glmb_out_n = glmb_update_n(idxkeep_eigen);
            VectorXd glmb_out_cdn(glmb_out_n.maxCoeff() + 1);
            vector<VectorXi> glmb_out_I;
            for (int i: idxkeep_eigen) {
                glmb_out_I.push_back(glmb_update_I[i]);
            }

            for (int card = 0; card < glmb_out_cdn.size(); card++) {
                glmb_out_cdn[card] = (glmb_out_n.array() == card).select(glmb_out_n, 0).sum();
            }
            glmb_update_w = glmb_out_w;  // 2
            glmb_update_I = glmb_out_I;  // 3
            glmb_update_n = glmb_out_n;  // 4
            glmb_update_cdn = glmb_out_cdn;  // 5
        }
    }

    std::tuple<MatrixXd, int, MatrixXi> extract_estimates(Model model) {
        // extract estimates via best cardinality, then
        // best component/hypothesis given best cardinality, then
        // best means of tracks given best component/hypothesis and cardinality
        int N;
        glmb_update_cdn.maxCoeff(&N);
        MatrixXd X(model.x_dim, N);
        MatrixXi L(2, N);
        int idxcmp;
        (glmb_update_w.array() * (glmb_update_n.array() == N).cast<double>()).maxCoeff(&idxcmp);
        for (int n = 0; n < N; n++) {
            int idxptr = glmb_update_I[idxcmp](n);
            X.col(n) = glmb_update_tt[idxptr].m;
            L.col(n) = glmb_update_tt[idxptr].l;
        }
        return {X, N, L};
    }

public:
    MSGLMB(vector<MatrixXd> camera_mat) {
        glmb_update_w = VectorXd::Ones(1);
        glmb_update_I.push_back(VectorXi(0));
        glmb_update_n = VectorXi::Zero(1);
        glmb_update_cdn = VectorXd::Ones(1);
        model = Model(camera_mat);
        filter = Filter();
    }

    std::tuple<MatrixXd, int, MatrixXi> run_msglmb_ukf(vector<MatrixXd> measZ, int kt) {

        msjointpredictupdate(model, filter, measZ, kt);
        int H_posterior = glmb_update_w.size();

        // pruning and truncation
        prune(filter);
        int H_prune = glmb_update_w.size();
        cap(filter);
        int H_cap = glmb_update_w.size();
        clean_update();

        VectorXd rangetmp = VectorXd::LinSpaced(glmb_update_cdn.size(), 0, glmb_update_cdn.size() - 1);
        cout << "Time " << kt << " #eap cdn=" << rangetmp.transpose() * glmb_update_cdn;
        int temp1 = ((VectorXd) rangetmp.array().pow(2)).transpose() * glmb_update_cdn;
        int temp2 = (rangetmp.transpose() * glmb_update_cdn);
        cout << " #var cdn=" << temp1 - temp2 * temp2;
        cout << " #comp pred=" << H_posterior;
        cout << " #comp post=" << H_posterior;
        cout << " #comp updt=" << H_cap;
        cout << " #trax updt=" << glmb_update_tt.size() << endl;

        // state estimation and display diagnostics
        return extract_estimates(model);
    }
};


#endif //UKF_TARGET_MS_GLMB_UKF_HPP
