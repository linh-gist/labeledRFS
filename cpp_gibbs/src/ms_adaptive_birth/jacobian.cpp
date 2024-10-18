//
// Created by linh on 2022-06-08.
//

#include <Eigen/Core>

using namespace Eigen;
using namespace std;

MatrixXd state2bbox(VectorXd x, const MatrixXd &cam_mat) {
    // gen_msobservation_fn_v2, approximation of using ellipsoid
    // x is extracted from [0, 1, 2, 3, 4, 5, 6, 7, 8] => [0, 2, 4, 6, 7, 8]
    MatrixXd vet2 = MatrixXd::Ones(4, 6);
    vet2.col(0) << x[0] + x[3], x[1], x[2], 1;  // -right
    vet2.col(1) << x[0] - x[3], x[1], x[2], 1;  // -left
    vet2.col(2) << x[0], x[1] + x[4], x[2], 1;  // | right
    vet2.col(3) << x[0], x[1] - x[4], x[2], 1;  // | left
    vet2.col(4) << x[0], x[1], x[2] + x[5], 1;
    vet2.col(5) << x[0], x[1], x[2] - x[5], 1;
    MatrixXd temp_c = cam_mat * vet2;
    MatrixXd vertices = temp_c(seq(0, 1), all).array().rowwise() / temp_c(2, all).array();
    return vertices;
}

double f_column(VectorXd x, MatrixXd cam_mat, int row_idx, int col_idx) {
    // row_idx: 0 or 1
    // col_idx: 0, 1, 2, 3, 4, 5
    VectorXd temp_row = cam_mat.row(row_idx);
    VectorXd temp_col(4);
    switch (col_idx) {
        case 0:
            temp_col << x[0] + x[3], x[1], x[2], 1;  // -right
            break;
        case 1:
            temp_col << x[0] - x[3], x[1], x[2], 1;  // -left
            break;
        case 2:
            temp_col << x[0], x[1] + x[4], x[2], 1;  // | right
            break;
        case 3:
            temp_col << x[0], x[1] - x[4], x[2], 1;  // | left
            break;
        case 4:
            temp_col << x[0], x[1], x[2] + x[5], 1;
            break;
        case 5:
            temp_col << x[0], x[1], x[2] - x[5], 1;
            break;
        default:
            break;
    }
    VectorXd cam_mat2 = cam_mat.row(2);
    double row2 = cam_mat2.transpose() * temp_col;
    double value = temp_row.transpose() * temp_col;
    return value / row2;
}

VectorXd func(const VectorXd &x, const MatrixXd &cam_mat) {
    MatrixXd vertices = state2bbox(x, cam_mat);
    int z0_col;
    vertices.row(0).minCoeff(&z0_col);
    int z1_col;
    vertices.row(1).minCoeff(&z1_col);
    int z2_col;
    vertices.row(0).maxCoeff(&z2_col);
    int z3_col;
    vertices.row(1).maxCoeff(&z3_col);
    double f0 = f_column(x, cam_mat, 0, z0_col);
    double f1 = f_column(x, cam_mat, 1, z1_col);
    double f2 = log(f_column(x, cam_mat, 0, z2_col) - f_column(x, cam_mat, 0, z0_col));
    double f3 = log(f_column(x, cam_mat, 1, z3_col) - f_column(x, cam_mat, 1, z1_col));
    VectorXd result(4);
    result << f0, f1, f2, f3;
    return result;
}

MatrixXd jacobian(const VectorXd &initial, const MatrixXd &cam_mat, double delta = 1e-3) {
    // https://rh8liuqy.github.io/Finite_Difference.html
    int nrow = func(initial, cam_mat).size();
    int ncol = initial.size();
    MatrixXd output = MatrixXd::Zero(nrow, ncol);
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            VectorXd ej = VectorXd::Zero(ncol);
            ej[j] = 1;
            double dij = (func(initial + delta * ej, cam_mat)[i] -
                          func(initial - delta * ej, cam_mat)[i]) / (2 * delta);
            output(i, j) = dij;
        }
    }
    return output;
}
