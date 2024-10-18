//
// Created by linh on 2022-03-31.
//

#ifndef UKF_TARGET_TARGET_HPP
#define UKF_TARGET_TARGET_HPP

#include "Model.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;
using namespace Eigen;


class Target {
private:

    Model mModel;

public:
    VectorXd m;
    MatrixXd P;
    vector<VectorXi> gatemeas;
    double r;
    double PS;
    double PD;
    Vector2i l;

    Target() {};

    Target(VectorXd m, MatrixXd P, double prob_birth, double PS, double PD, Vector2i label, Model model) {
        this->m = m;
        this->P = P;
        this->r = prob_birth;
        this->PS = PS;
        this->PD = PD;
        this->l = label;
        this->mModel = model;
        this->gatemeas.resize(mModel.N_sensors);
    }

    void predict() {
        // this is to offset the normal mean because of lognormal multiplicative noise.
        VectorXd offset = VectorXd::Zero(mModel.x_dim);
        offset(seq(6, 8)) << mModel.n_mu(0), mModel.n_mu(0), mModel.n_mu(1);
        VectorXd m_per_mode = m + offset;
        m = mModel.F * m_per_mode;
        P = mModel.Q + mModel.F * P * mModel.F.transpose();
    }

    double ukf_update_per_sensor(VectorXd z, int s, double alpha, double kappa, double beta) {
        bool ch1 = m(0) > mModel.XMAX(0) && m(0) < mModel.XMAX(1);
        bool ch2 = m(2) > mModel.YMAX(0) && m(2) < mModel.YMAX(1);
        bool ch3 = m(4) > mModel.ZMAX(0) && m(4) < mModel.ZMAX(1);
        if (!(ch1 && ch2 && ch3)) {
            return log(std::nexttoward(0.0, 1.0L));  // qz_temp
        }
        VectorXd mtemp = VectorXd::Zero(mModel.x_dim + mModel.z_dim);
        mtemp(seq(0, mModel.x_dim - 1)) = m;
        MatrixXd Ptemp = MatrixXd::Zero(mModel.x_dim + mModel.z_dim, mModel.x_dim + mModel.z_dim);
        Ptemp.topLeftCorner(mModel.x_dim, mModel.x_dim) = P;
        Ptemp.bottomRightCorner(mModel.z_dim, mModel.z_dim) = mModel.R;
        MatrixXd X_ukf;
        VectorXd u;
        std::tie(X_ukf, u) = ut(mtemp, Ptemp, alpha, kappa);
        VectorXd temp(4);
        temp << 0, 0, mModel.meas_n_mu(0), mModel.meas_n_mu(1);
        int start = mModel.x_dim;
        int end = mModel.x_dim + mModel.z_dim - 1;
        X_ukf(seq(start, end), all) = X_ukf(seq(start, end), all).colwise() + temp;
        MatrixXd Z_pred = gen_msobservation_fn(mModel.camera_mat[s], X_ukf(seq(0, start - 1), all),
                                               X_ukf(seq(start, end), all), mModel.imagesize);
        VectorXd eta = Z_pred * u;
        MatrixXd S_temp = Z_pred.colwise() - eta;
        u(0) = u(0) + (1 - alpha * alpha + beta);
        MatrixXd S = S_temp * u.asDiagonal() * S_temp.transpose();
        MatrixXd Vs = S.llt().matrixU();
        double det_S = Vs.diagonal().prod();
        det_S = det_S * det_S;
        MatrixXd inv_sqrt_S = Vs.inverse();
        MatrixXd iS = inv_sqrt_S * inv_sqrt_S.transpose();
        // MatrixXd G_temp = X_ukf(seq(0, mModel.x_dim), all).colwise() - mM;
        // MatrixXd G = G_temp * u.asDiagonal() * S_temp.transpose();
        // MatrixXd K = G * iS;
        VectorXd z_eta = z - eta;
        double qz_temp = -0.5 * (z.size() * log(2 * M_PI) + log(det_S) + z_eta.transpose() * iS * z_eta);
        qz_temp = exp(qz_temp);
        // VectorXd m_temp = mM + K * z_eta;
        // MatrixXd P_temp = mP - G * iS * G.transpose();

        return qz_temp;
    }

    void ukf_msjointupdate(vector<MatrixXd> Z, VectorXi nestmeasidxs, double alpha, double kappa, double beta,
                           Target &tt, double &qz_temp) {
        VectorXi slogidxs = (nestmeasidxs.array() > 0).cast<int>();
        if (slogidxs.sum() == 0) {
            qz_temp = 1;
            tt = Target(m, P, r, PS, PD, l, mModel);
            return;
        }
        vector<MatrixXd> stacked_R;
        vector<VectorXd> stacked_z;
        nestmeasidxs = nestmeasidxs.array() - 1; // restore original measurement index 0-|Z|
        int s_multi = 0;
        for (int idx = 0; idx < slogidxs.size(); idx++) {
            if (slogidxs(idx)) {
                stacked_R.push_back(mModel.R);
                stacked_z.push_back(Z[idx].col(nestmeasidxs(idx)));
                s_multi += mModel.z_dim;
            }
        }
        VectorXd newZ(s_multi);
        MatrixXd newR = MatrixXd::Zero(s_multi, s_multi);
        for (int idx = 0; idx < stacked_z.size(); idx++) {
            int start = idx * mModel.z_dim;
            int end = (idx + 1) * mModel.z_dim;
            newZ(seq(start, end - 1)) = stacked_z[idx];
            newR(seq(start, end - 1), seq(start, end - 1)) = stacked_R[idx];
        }
        VectorXd mtemp = VectorXd::Zero(mModel.x_dim + s_multi);
        mtemp(seq(0, mModel.x_dim - 1)) = m;
        MatrixXd Ptemp = MatrixXd::Zero(mModel.x_dim + s_multi, mModel.x_dim + s_multi);
        Ptemp.topLeftCorner(mModel.x_dim, mModel.x_dim) = P;
        Ptemp.bottomRightCorner(s_multi, s_multi) = newR;
        MatrixXd X_ukf;
        VectorXd u;
        std::tie(X_ukf, u) = ut(mtemp, Ptemp, alpha, kappa);
        int start_indx = mModel.x_dim;
        MatrixXd Z_pred = MatrixXd::Zero(4 * slogidxs.sum(), X_ukf.cols());
        int z_idx = 0;
        for (int idx = 0; idx < slogidxs.size(); idx++) {
            if (slogidxs(idx)) {
                int end_indx = start_indx + mModel.z_dim;
                VectorXd temp(4);
                temp << 0, 0, mModel.meas_n_mu(0), mModel.meas_n_mu(1);
                MatrixXd tempX = X_ukf(seq(start_indx, end_indx - 1), all).colwise() + temp;
                X_ukf(seq(start_indx, end_indx - 1), all) = tempX;
                tempX = X_ukf(seq(0, mModel.x_dim - 1), all);
                MatrixXd tempW = X_ukf(seq(start_indx, end_indx - 1), all);
                MatrixXd Z_temp = gen_msobservation_fn(mModel.camera_mat[idx], tempX, tempW, mModel.imagesize);
                start_indx = end_indx;
                Z_pred(seq(4 * z_idx, 4 * (z_idx + 1) - 1), all) = Z_temp;
                z_idx += 1;
            }
        }
        VectorXd eta = Z_pred * u;
        MatrixXd S_temp = Z_pred.colwise() - eta;
        u(0) = u(0) + (1 - alpha * alpha + beta);
        MatrixXd S = S_temp * u.asDiagonal() * S_temp.transpose();
        MatrixXd Vs = S.llt().matrixU();
        double det_S = Vs.diagonal().prod();
        det_S = det_S * det_S;
        MatrixXd inv_sqrt_S = Vs.inverse();
        MatrixXd iS = inv_sqrt_S * inv_sqrt_S.transpose();
        MatrixXd G_temp = X_ukf(seq(0, mModel.x_dim - 1), all).colwise() - m;
        MatrixXd G = G_temp * u.asDiagonal() * S_temp.transpose();
        MatrixXd K = G * iS;
        VectorXd z_eta = newZ - eta;
        qz_temp = -0.5 * (newZ.size() * log(2 * M_PI) + log(det_S) + (newZ - eta).transpose() * iS * (newZ - eta));
        qz_temp = exp(qz_temp);
        VectorXd m_temp = m + K * z_eta;
        MatrixXd P_temp = P - G * iS * G.transpose();

        // deep copy this Target to a new Target
        tt = Target(m_temp, P_temp, r, PS, PD, l, mModel);
        //return {qz_temp, tt};
    }

    void gate_msmeas_ukf(MatrixXd Zs, int s, double gamma, double alpha, double kappa, double beta) {
        int zlength = Zs.cols();
        if (zlength == 0) {
            gatemeas[s] = VectorXi(0);
        }
        VectorXd mtemp = VectorXd::Zero(mModel.x_dim + mModel.z_dim);
        mtemp(seq(0, mModel.x_dim - 1)) = m;
        MatrixXd Ptemp = MatrixXd::Zero(mModel.x_dim + mModel.z_dim, mModel.x_dim + mModel.z_dim);
        Ptemp.topLeftCorner(mModel.x_dim, mModel.x_dim) = P;
        Ptemp.bottomRightCorner(mModel.z_dim, mModel.z_dim) = mModel.R;
        MatrixXd X_ukf;
        VectorXd u;
        std::tie(X_ukf, u) = ut(mtemp, Ptemp, alpha, kappa);
        MatrixXd tempX = X_ukf(seq(0, mModel.x_dim - 1), all);
        MatrixXd tempW = X_ukf(seq(mModel.x_dim, mModel.x_dim + mModel.z_dim - 1), all);
        MatrixXd Z_pred = gen_msobservation_fn(mModel.camera_mat[s], tempX, tempW, mModel.imagesize);
        VectorXd eta = Z_pred * u;
        MatrixXd Sj_temp = Z_pred.colwise() - eta;
        u(0) = u(0) + (1 - alpha * alpha + beta);
        MatrixXd Sj = Sj_temp * u.asDiagonal() * Sj_temp.transpose();
        MatrixXd Vs = Sj.llt().matrixU();
        // double det_Sj = Vs.diagonal().prod();
        // det_Sj = det_Sj * det_Sj;
        MatrixXd inv_sqrt_Sj = Vs.inverse();
        // MatrixXd iSj = inv_sqrt_Sj * inv_sqrt_Sj.transpose();
        VectorXd w_noise = VectorXd::Zero(mModel.z_dim);
        VectorXd m_z = gen_msobservation_fn(mModel.camera_mat[s], m, w_noise, mModel.imagesize);
        MatrixXd nu = Zs.colwise() - m_z;
        MatrixXd dist_temp = inv_sqrt_Sj.transpose() * nu;
        dist_temp = dist_temp.array() * dist_temp.array();
        VectorXd dist = dist_temp.colwise().sum();
        VectorXi dist_gate = (dist.array() < gamma).cast<int>();
        gatemeas[s] = -1 * VectorXi::Ones(zlength);
        for (int i = 0; i < zlength; i++) {
            if (dist_gate(i)) {
                gatemeas[s](i) = i; // only gated measurements have non-negative values
            }
        }
    }

    void not_gating(vector<MatrixXd> Zz) {
        gatemeas.resize(mModel.N_sensors);
        for (int s = 0; s < mModel.N_sensors; s++) {
            gatemeas[s] = VectorXi::LinSpaced(Zz[s].cols(), 0, Zz[s].cols() - 1);
        }
    }

};


#endif //UKF_TARGET_TARGET_HPP
