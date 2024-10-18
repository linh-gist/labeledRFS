import numpy as np


class model:
    def __init__(self):
        # basic parameters
        self.x_dim = 4;  # dimension of state vector
        self.z_dim = 2;  # dimension of observation vector

        # dynamical model parameters (CV model)
        self.T = 1;  # sampling period
        self.A0 = np.array([[1, self.T], [0, 1]]);  # transition matrix
        self.F = np.concatenate((np.concatenate((self.A0, np.zeros((2, 2))), axis=1),
                                 np.concatenate((np.zeros((2, 2)), self.A0), axis=1))
                                , axis=0);
        self.B0 = np.array([[(self.T ** 2) / 2], [self.T]]);
        self.B = np.concatenate((np.concatenate((self.B0, np.zeros((2, 1))), axis=1),
                                 np.concatenate((np.zeros((2, 1)), self.B0), axis=1))
                                , axis=0);
        self.sigma_v = 5;
        self.Q = (self.sigma_v) ** 2 * np.dot(self.B, self.B.T);  # process noise covariance

        # survival/death parameters
        self.P_S = .99;
        self.Q_S = 1 - self.P_S;

        # birth parameters (LMB birth model, single component only)
        self.T_birth = 4;  # no. of LMB birth terms
        self.L_birth = np.zeros((self.T_birth, 1));  # no of Gaussians in each LMB birth term
        self.r_birth = np.zeros((self.T_birth, 1));  # prob of birth for each LMB birth term
        self.w_birth = np.zeros((self.T_birth, 1));  # weights of GM for each LMB birth term
        self.m_birth = np.zeros((self.T_birth, self.T_birth, 1));  # means of GM for each LMB birth term
        self.B_birth = np.zeros((self.T_birth, self.T_birth, self.T_birth));  # std of GM for each LMB birth term
        self.P_birth = np.zeros((self.T_birth, self.T_birth, self.T_birth));  # cov of GM for each LMB birth term

        self.L_birth[0] = 1;  # no of Gaussians in birth term 1
        self.r_birth[0] = 0.03;  # prob of birth
        self.w_birth[0] = 1;  # weight of Gaussians - must be column_vector
        self.m_birth[0, :, :] = np.array([[0.1], [0], [0.1], [0]]);  # mean of Gaussians
        self.B_birth[0, :, :] = np.diagflat([[10], [10], [10], [10]]);  # std of Gaussians
        self.P_birth[0, :, :] = self.B_birth[0][:, :] * self.B_birth[0][:, :].T;  # cov of Gaussians

        self.L_birth[1] = 1;  # no of Gaussians in birth term 2
        self.r_birth[1] = 0.03;  # prob of birth
        self.w_birth[1] = 1;  # %weight of Gaussians - must be column_vector
        self.m_birth[1, :, :] = np.array([[400], [0], [-600], [0]]);  # mean of Gaussians
        self.B_birth[1, :, :] = np.diagflat([[10], [10], [10], [10]]);  # std of Gaussians
        self.P_birth[1, :, :] = self.B_birth[0][:, :] * self.B_birth[0][:, :].T;  # cov of Gaussians

        self.L_birth[2] = 1;  # no of Gaussians in birth term 3
        self.r_birth[2] = 0.03;  # prob of birth
        self.w_birth[2] = 1;  # weight of Gaussians - must be column_vector
        self.m_birth[2, :, :] = np.array([[-800], [0], [-200], [0]]);  # mean of Gaussians
        self.B_birth[2, :, :] = np.diagflat([[10], [10], [10], [10]]);  # std of Gaussians
        self.P_birth[2, :, :] = self.B_birth[0][:, :] * self.B_birth[0][:, :].T;  # cov of Gaussians

        self.L_birth[3] = 1;  # no of Gaussians in birth term 4
        self.r_birth[3] = 0.03;  # prob of birth
        self.w_birth[3] = 1;  # weight of Gaussians - must be column_vector
        self.m_birth[3, :, :] = [[-200], [0], [800], [0]];  # mean of Gaussians
        self.B_birth[3, :, :] = np.diagflat([[10], [10], [10], [10]]);  # std of Gaussians
        self.P_birth[3, :, :] = self.B_birth[0][:, :] * self.B_birth[0][:, :].T;  # cov of Gaussians

        # observation model parameters (noisy x/y only)
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]]);  # observation matrix
        self.D = np.diag([10, 10]);
        self.R = self.D * self.D.T;  # observation noise covariance

        # detection parameters
        self.P_D = .98;  # probability of detection in measurements
        self.Q_D = 1 - self.P_D;  # probability of missed detection in measurements

        # clutter parameters
        self.lambda_c = 30;  # poisson average rate of uniform clutter (per scan)
        self.range_c = np.array([[-1000, 1000], [-1000, 1000]]);  # uniform clutter region
        self.pdf_c = 1 / np.prod(self.range_c[:, 1] - self.range_c[:, 0]);  # uniform clutter density


if __name__ == '__main__':
    model_params = model()
    print(model_params)
