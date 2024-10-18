//
// Created by linh on 2022-04-01.
//

#ifndef UKF_TARGET_MODEL_HPP
#define UKF_TARGET_MODEL_HPP

#include <vector>
#include <Eigen/Dense>
#include <Eigen/LU>  /* MatrixBase::inversemethode is definded in the Eigen::LU */
#include <cmath>
#include <iostream>

using namespace std;
using namespace Eigen;

tuple<MatrixXd, VectorXd> ut(VectorXd m, MatrixXd P, double alpha, double kappa) {
    int n_x = m.size();
    double lambda_ = alpha * alpha * (n_x + kappa) - n_x;
    MatrixXd Psqrtm = ((n_x + lambda_) * P).llt().matrixL(); // lower-triangular Cholesky
    MatrixXd temp = MatrixXd::Zero(n_x, 2 * n_x + 1);
    temp(all, seq(1, n_x)) = -Psqrtm;
    temp(all, seq(n_x + 1, 2 * n_x)) = Psqrtm;
    MatrixXd X = temp.colwise() + m;
    VectorXd W = VectorXd::Ones(2 * n_x + 1);
    W = 0.5 * W;
    W(0) = lambda_;
    W = W / (n_x + lambda_);
    return {X, W};
}

MatrixXd gen_msobservation_fn(MatrixXd cam_mat, MatrixXd X, MatrixXd W, Vector2d imagesize) {
    if (X.rows() == 0) {
        return MatrixXd(0, 0);
    }
    X(seq(6,8), all) = X(seq(6,8), all).array().exp();  // Carefully check reference or pointer
    MatrixXd bbs_noiseless(4, X.cols());
    for (int i = 0; i < X.cols(); i++) {
        VectorXd ellipsoid_c(3);
        ellipsoid_c << X(0, i), X(2, i), X(4, i); // xc, yc, zc
        // Quadric general equation 0 = Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz + J
        // Q = [A D/2 E/2 G/2;
        //      D/2 B F/2 H/2;
        //      E/2 F/2 C I/2;
        //      G/2 H/2 I/2 J];
        // calculations for A, B, C, rx, ry, hh = X[6, i], X[7, i], X[8, i]  # half length radius (x, y, z)
        double A = 1 / (X(6, i) * X(6, i));
        double B = 1 / (X(7, i) * X(7, i));
        double C = 1 / (X(8, i) * X(8, i));
        // calculations for D, E, F, no rotation (axis-aligned) means D, E, F = 0
        double D = 0;
        double E = 0;
        double F = 0;
        // calculations for G, H, I, J
        MatrixXd PSD(3, 3); // np.diag([A, B, C])
        PSD.setZero();
        PSD(0, 0) = A;
        PSD(1, 1) = B;
        PSD(2, 2) = C;
        EigenSolver<MatrixXd> es(PSD); // This constructor calls compute() to compute the values and vectors.
        MatrixXd eig_vals = es.eigenvalues().real();
        MatrixXd right_eig = es.eigenvectors().real(); // [V,D] = eig(A), right eigenvectors, so that A*V = V*D
        VectorXd temp_ellip_c = right_eig.transpose() * ellipsoid_c;
        VectorXd ggs = -2 * temp_ellip_c.array() * eig_vals.array();
        VectorXd desired = ggs.transpose() * right_eig;
        double G = desired[0];
        double H = desired[1];
        double I = desired[2];
        double J = -1 + (ggs.array() * ggs.array() / (4 * eig_vals.array())).sum();
        MatrixXd Q(4, 4);
        Q << A, D / 2, E / 2, G / 2,
                D / 2, B, F / 2, H / 2,
                E / 2, F / 2, C, I / 2,
                G / 2, H / 2, I / 2, J; // 4x4 matrix
        MatrixXd C_t = cam_mat * Q.inverse() * cam_mat.transpose();
        MatrixXd CI = C_t.inverse(); // 3x3 matrix
        MatrixXd C_strip = CI(seq(0, 1), seq(0, 1));
        es.compute(C_strip);
        eig_vals = es.eigenvalues().real();
        right_eig = es.eigenvectors().real(); // [V,D] = eig(A), right eigenvectors, so that A*V = V*D
        VectorXd x_and_y_vec = 2 * CI(seq(0, 1), 2); // extrack D and E
        VectorXd x_and_y_vec_transformed = x_and_y_vec.transpose() * right_eig;
        VectorXd h_temp = (x_and_y_vec_transformed.array() / eig_vals.array()) / 2;
        VectorXd h_temp_squared = eig_vals.array() * (h_temp.array() * h_temp.array());
        VectorXd h = -1 * h_temp;
        VectorXd ellipse_c = right_eig * h;
        double offset = -1 * (h_temp_squared.sum()) + CI(2, 2);
        VectorXd bbs_temp(4);
        if ((-offset / eig_vals(0) > 0) && (-offset / eig_vals(1) > 0)) {
            VectorXd uu = right_eig.col(0) * sqrt(-offset / eig_vals(0));
            VectorXd vv = right_eig.col(1) * sqrt(-offset / eig_vals(1));
            VectorXd e = (uu.array() * uu.array() + vv.array() * vv.array()).sqrt();
            MatrixXd bbox(2, 2);
            bbox.col(0) = ellipse_c - e;
            bbox.col(1) = ellipse_c + e;

            double tl0 = bbox.row(0).minCoeff(); // top_left0
            double tl1 = bbox.row(1).minCoeff(); // top_left1
            double br0 = bbox.row(0).maxCoeff(); // bottm_right0
            double br1 = bbox.row(1).maxCoeff();//  bottm_right1

            bbs_temp << tl0, tl1, log((br0 - tl0)), log((br1 - tl1));
        } else {
            // top_left = [1 1];
            // bottm_right = imagesize
            bbs_temp << 1, 1, log((imagesize(0) - 1)), log((imagesize(1) - 1));
        }
        bbs_noiseless.col(i) = bbs_temp;
    }
    return bbs_noiseless + W;  // bounding measurement
}

Vector2d lognormal_with_mean_one(double percen) {
    // input is std dev of multiplicative lognormal noise.
    Vector2d mean_std;
    double percen_v = percen * percen;
    mean_std(1) = sqrt(log(percen_v + 1));
    mean_std(0) = -mean_std(1) * mean_std(1) / 2;
    return mean_std;
}

class Model {
public:
    MatrixXd F;
    MatrixXd Q;
    MatrixXd R;
    Vector2d XMAX;
    Vector2d YMAX;
    Vector2d ZMAX;
    int x_dim;
    int z_dim;
    Vector2d imagesize;
    Vector2d meas_n_mu;
    Vector2d n_mu;
    vector<MatrixXd> camera_mat;
    int N_sensors;

    vector<VectorXd> m;
    MatrixXd P;
    double lambda_c;
    double pdf_c;

    Model() {};

    Model(vector<MatrixXd> camera_mat) {
        x_dim = 9;
        z_dim = 4;
        XMAX << 2.03, 6.3; // 2.03 5.77 6.3
        YMAX << 0.00, 3.41; // [0.05 3.41];
        ZMAX << 0, 3; // 5.77
        imagesize << 1920, 1024;
        N_sensors = 4;

        F.resize(9, 9);
        F << 1., 1., 0., 0., 0., 0., 0., 0., 0.,
                0., 1., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 1., 1., 0., 0., 0., 0., 0.,
                0., 0., 0., 1., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 1., 1., 0., 0., 0.,
                0., 0., 0., 0., 0., 1., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 1., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 1., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 1.;

        double T = 1.0; // sampling period
        double sigma_v = 0.005;
        Vector2d B0;
        B0 << T * T / 2.0, T;
        B0 = sigma_v * B0;
        Vector2d n_mu_std0 = lognormal_with_mean_one(0.06);
        Vector2d n_mu_std1 = lognormal_with_mean_one(0.02);
        MatrixXd B1 = MatrixXd::Zero(3, 3);
        B1(0, 0) = n_mu_std0(1); // sigma_radius
        B1(1, 1) = n_mu_std0(1); // sigma_radius
        B1(2, 2) = n_mu_std1(1); // sigma_heig
        Matrix3d I3 = Matrix3d::Identity();
        MatrixXd P2(I3.rows() * B0.rows(), I3.cols() * B0.cols());
        P2.setZero();
        for (int i = 0; i < I3.RowsAtCompileTime; i++) { // Kronecker product of two arrays I3*B0
            P2.block(i * B0.rows(), i * B0.cols(), B0.rows(), B0.cols()) = I3(i, i) * B0;
        }
        MatrixXd B = MatrixXd::Zero(9, 6);
        B.topLeftCorner(6, 3) = P2;
        B.bottomRightCorner(3, 3) = B1;
        Q = B * B.transpose();

        Vector2d meas_mu_std0 = lognormal_with_mean_one(0.1);
        Vector2d meas_mu_std1 = lognormal_with_mean_one(0.05);
        VectorXd D_temp(4);
        D_temp << 5, 5, meas_mu_std0(1), meas_mu_std1(1);
        MatrixXd D = D_temp.asDiagonal();
        R = D * D.transpose();

        n_mu << n_mu_std0(0), n_mu_std1(0);
        meas_n_mu << meas_mu_std0(0), meas_mu_std1(0);

        this->camera_mat = camera_mat;

        m.resize(4);
        Vector2d n_mu_hold = lognormal_with_mean_one(0.1);
        m[0].resize(9);
        m[0] << 2.52, 0, 0.71, 0, 0.825, 0, log(0.3) + n_mu_hold(0), log(0.3) + n_mu_hold(0), log(0.84) + n_mu_hold(0);
        m[1].resize(9);
        m[1] << 2.52, 0.0, 2.20, 0, 0.825, 0, log(0.3) + n_mu_hold(0), log(0.3) + n_mu_hold(0), log(0.84) + n_mu_hold(0);
        m[2].resize(9);
        m[2] << 5.5, 0, 2.20, 0, 0.825, 0, log(0.3) + n_mu_hold(0), log(0.3) + n_mu_hold(0), log(0.84) + n_mu_hold(0);
        m[3].resize(9);
        m[3] << 5.5, 0, 0.71, 0, 0.825, 0, log(0.3) + n_mu_hold(0), log(0.3) + n_mu_hold(0), log(0.84) + n_mu_hold(0);

        VectorXd b(9);
        b << 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, n_mu_hold(1), n_mu_hold(1), n_mu_hold(1);
        MatrixXd diag_b = b.asDiagonal();
        P = diag_b * diag_b.transpose();
        lambda_c = 10;
        pdf_c = 1 / (1920 * 1024 * log(1920) * log(1024));
    };
};

#endif //UKF_TARGET_MODEL_HPP
