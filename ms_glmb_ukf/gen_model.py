import numpy as np
from scipy.linalg import block_diag
import h5py


def lognormal_with_mean_one(percen):
    percen_v = percen ** 2
    std_dev = np.sqrt(np.log(percen_v + 1))
    mean = - std_dev ** 2 / 2
    return mean, std_dev


class model:
    def __init__(self):
        # basic parameters
        self.x_dim = 9  # dimension of state vector
        self.XMAX = [2.03, 6.3]  # 2.03 5.77 6.3
        self.YMAX = [0.00, 3.41]  # [0.05 3.41];
        self.ZMAX = [0, 3]  # 5.77

        # param for ellipsoid plotting
        self.ellipsoid_n = 10

        # dynamical model parameters (CV model)
        self.T = 1  # sampling period
        self.A0 = np.array([[1, self.T],
                            [0, 1]])  # transition matrix
        self.F = block_diag(*[np.kron(np.eye(3, dtype='f8'), self.A0), np.eye(3, dtype='f8')])
        self.sigma_v = 0.005  # 0.035
        n_mu0, n_std_dev0 = lognormal_with_mean_one(0.06)  # input is std dev of multiplicative lognormal noise.
        n_mu1, n_std_dev1 = lognormal_with_mean_one(0.02)
        self.n_mu, self.n_std_dev = np.array([n_mu0, n_mu1]),  np.array([n_std_dev0, n_std_dev1])
        self.sigma_radius = n_std_dev0
        self.sigma_heig = n_std_dev1
        self.B0 = self.sigma_v * np.array([[(self.T ** 2) / 2], [self.T]])
        self.B1 = np.diag([self.sigma_radius, self.sigma_radius, self.sigma_heig])
        self.B = block_diag(*[np.kron(np.eye(3, dtype='f8'), self.B0), self.B1])
        self.Q = np.dot(self.B, self.B.T)  # process noise covariance

        # survival/death parameters
        self.P_S = .99
        self.Q_S = 1 - self.P_S

        # birth parameters (LMB birth model, single component only)
        self.T_birth = 4  # no. of LMB birth terms
        self.L_birth = np.zeros((self.T_birth, 1))  # no of Gaussians in each LMB birth term
        self.r_birth = np.zeros(self.T_birth)  # prob of birth for each LMB birth term
        self.w_birth = np.zeros((self.T_birth, 1))  # weights of GM for each LMB birth term
        self.m_birth = np.zeros((self.T_birth, self.x_dim, 1))  # means of GM for each LMB birth term
        self.B_birth = np.zeros((self.T_birth, self.x_dim, self.x_dim))  # std of GM for each LMB birth term
        self.P_birth = np.zeros((self.T_birth, self.x_dim, self.x_dim))  # cov of GM for each LMB birth term

        self.L_birth[0] = 1  # no of Gaussians in birth term 1
        self.r_birth[0] = 0.0001  # 0.0001 Use for J_5_2 % 0.001 use for J_6_1      00006
        self.w_birth[0] = 1  # weight of Gaussians - must be column_vector
        n_mu_hold, n_std_dev_hold = lognormal_with_mean_one(0.1)  # input is std dev of multiplicative lognormal noise.
        self.m_birth[0, :, :] = np.array([[2.52], [0], [0.71], [0], [0.825], [0], [np.log(0.3) + n_mu_hold],
                                          [np.log(0.3) + n_mu_hold], [np.log(0.84) + n_mu_hold]])  # mean of Gaussians
        self.B_birth[0, :, :] = np.diagflat([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, n_std_dev_hold, n_std_dev_hold,
                                             n_std_dev_hold])  # std of Gaussians
        self.P_birth[0, :, :] = np.dot(self.B_birth[0], self.B_birth[0].T)  # cov of Gaussians

        self.L_birth[1] = 1  # no of Gaussians in birth term 2
        self.r_birth[1] = 0.0001  # 0.0001 Use for J_5_2 % 0.001 use for J_6_1      00006
        self.w_birth[1] = 1  # %weight of Gaussians - must be column_vector
        self.m_birth[1, :, :] = np.array([[2.52], [0.0], [2.20], [0], [0.825], [0], [np.log(0.3) + n_mu_hold],
                                          [np.log(0.3) + n_mu_hold], [np.log(0.84) + n_mu_hold]])  # mean of Gaussians
        self.B_birth[1, :, :] = np.diagflat([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, n_std_dev_hold, n_std_dev_hold,
                                             n_std_dev_hold])  # std of Gaussians
        self.P_birth[1, :, :] = np.dot(self.B_birth[1], self.B_birth[1].T)  # cov of Gaussians

        self.L_birth[2] = 1  # no of Gaussians in birth term 3
        self.r_birth[2] = 0.0001  # 0.0001 Use for J_5_2 % 0.001 use for J_6_1      00006
        self.w_birth[2] = 1  # weight of Gaussians - must be column_vector
        self.m_birth[2, :, :] = np.array([[5.5], [0], [2.20], [0], [0.825], [0], [np.log(0.3) + n_mu_hold],
                                          [np.log(0.3) + n_mu_hold], [np.log(0.84) + n_mu_hold]])  # mean of Gaussians
        self.B_birth[2, :, :] = np.diagflat([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, n_std_dev_hold, n_std_dev_hold,
                                             n_std_dev_hold])  # std of Gaussians
        self.P_birth[2, :, :] = np.dot(self.B_birth[2], self.B_birth[2].T)  # cov of Gaussians

        self.L_birth[3] = 1  # no of Gaussians in birth term 4
        self.r_birth[3] = 0.0001  # 0.0001 Use for J_5_2 % 0.001 use for J_6_1      00006
        self.w_birth[3] = 1  # weight of Gaussians - must be column_vector
        self.m_birth[3, :, :] = np.array([[5.5], [0], [0.71], [0], [0.825], [0], [np.log(0.3) + n_mu_hold],
                                          [np.log(0.3) + n_mu_hold], [np.log(0.84) + n_mu_hold]])  # mean of Gaussians
        self.B_birth[3, :, :] = np.diagflat([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, n_std_dev_hold, n_std_dev_hold,
                                             n_std_dev_hold])  # std of Gaussians
        self.P_birth[3, :, :] = np.dot(self.B_birth[3], self.B_birth[3].T)  # cov of Gaussians

        # multi-sensor observation parameters
        n_dim = 4  # noise_dim
        self.N_sensors = 4  # no of sensors
        self.z_dim = np.zeros(self.N_sensors, dtype=int)  # dimensions of observation vector for each sensor
        self.H = np.zeros((self.N_sensors, n_dim, self.x_dim))  # observation matrix for each sensor
        self.D = np.zeros((self.N_sensors, n_dim, n_dim))  # observation noise std for each sensor
        self.R = np.zeros((self.N_sensors, n_dim, n_dim))  # observation noise covariance for each sensor
        self.P_D = np.zeros((self.N_sensors, 1))  # probability of detection for each sensor
        self.Q_D = np.zeros((self.N_sensors, 1))  # probability of missed for each sensor
        self.lambda_c = np.zeros((self.N_sensors, 1))  # poisson clutter rate for each sensor
        self.range_c = np.zeros((self.N_sensors, n_dim, 2))  # uniform clutter region for each sensor
        self.pdf_c = np.zeros((self.N_sensors, 1))  # uniform clutter density for each sensor
        self.meas_n_mu = np.zeros((self.N_sensors, 2))
        self.meas_n_std_dev = np.zeros((self.N_sensors, 2))
        self.imagesize = [1920, 1024]
        self.cam_mat = np.zeros((self.N_sensors, 3, 4))

        # Sensor/Camera 1
        self.cam_mat[0] = h5py.File("cam1_cam_mat.mat").get("cam1_cam_mat").value.T
        self.z_dim[0] = 4
        self.lambda_c[0] = 10
        self.range_c[0] = np.array([[1, 1920], [1, 1024], [1, 1920], [1, 1024]])
        range_temp = self.range_c[0][:, 1] - self.range_c[0][:, 0] + 1
        range_temp[2: 4] = np.log(range_temp[2: 4])
        self.pdf_c[0] = 1 / np.prod(range_temp)
        self.P_D[0] = .98
        self.Q_D[0] = 1 - self.P_D[0]
        meas_n_mu0, meas_n_std_dev0 = lognormal_with_mean_one(0.1)  # 0.12
        meas_n_mu1, meas_n_std_dev1 = lognormal_with_mean_one(0.05)  # 0.07
        self.meas_n_mu[0, :] = [meas_n_mu0, meas_n_mu1]
        self.meas_n_std_dev[0, :] = [meas_n_std_dev0, meas_n_std_dev1]
        self.D[0] = np.diag([5, 5, meas_n_std_dev0, meas_n_std_dev1])  # diag([30;30;30;30])
        self.R[0] = np.dot(self.D[0], self.D[0].T)

        self.cam_mat[1] = h5py.File("cam2_cam_mat.mat").get("cam2_cam_mat").value.T
        # self.cam1_homo = [cam1_cam_mat[:, 0:2], cam1_cam_mat[:, 3]]
        self.z_dim[1] = 4
        self.lambda_c[1] = 10
        self.range_c[1] = np.array([[1, 1920], [1, 1024], [1, 1920], [1, 1024]])
        range_temp = self.range_c[1][:, 1] - self.range_c[1][:, 0] + 1
        range_temp[2: 4] = np.log(range_temp[2: 4])
        self.pdf_c[1] = 1 / np.prod(range_temp)
        self.P_D[1] = .98
        self.Q_D[1] = 1 - self.P_D[1]
        self.meas_n_mu[1, :] = [meas_n_mu0, meas_n_mu1]
        self.meas_n_std_dev[1, :] = [meas_n_std_dev0, meas_n_std_dev1]
        self.D[1] = np.diag([5, 5, meas_n_std_dev0, meas_n_std_dev1])  # diag([30;30;30;30])
        self.R[1] = np.dot(self.D[1], self.D[1].T)

        self.cam_mat[2] = h5py.File("cam3_cam_mat.mat").get("cam3_cam_mat").value.T
        # self.cam2_homo = [self.cam2_cam_mat[:, 0:2], self.cam2_cam_mat[:, 3]]
        self.z_dim[2] = 4
        self.lambda_c[2] = 10
        self.range_c[2] = np.array([[1, 1920], [1, 1024], [1, 1920], [1, 1024]])
        range_temp = self.range_c[2][:, 1] - self.range_c[2][:, 0] + 1
        range_temp[2: 4] = np.log(range_temp[2: 4])
        self.pdf_c[2] = 1 / np.prod(range_temp)
        self.P_D[2] = .98
        self.Q_D[2] = 1 - self.P_D[2]
        self.meas_n_mu[2, :] = [meas_n_mu0, meas_n_mu1]
        self.meas_n_std_dev[2, :] = [meas_n_std_dev0, meas_n_std_dev1]
        self.D[2] = np.diag([5, 5, meas_n_std_dev0, meas_n_std_dev1])  # diag([30;30;30;30])
        self.R[2] = np.dot(self.D[2], self.D[2].T)

        self.cam_mat[3] = h5py.File("cam4_cam_mat.mat").get("cam4_cam_mat").value.T
        # self.cam3_homo = [self.cam3_cam_mat[:, 0:2], self.cam3_cam_mat[:, 3]]
        self.z_dim[3] = 4
        self.lambda_c[3] = 10
        self.range_c[3] = np.array([[1, 1920], [1, 1024], [1, 1920], [1, 1024]])
        range_temp = self.range_c[3][:, 1] - self.range_c[3][:, 0] + 1
        range_temp[2: 4] = np.log(range_temp[2: 4])
        self.pdf_c[3] = 1 / np.prod(range_temp)
        self.P_D[3] = .98
        self.Q_D[3] = 1 - self.P_D[3]
        self.meas_n_mu[3, :] = [meas_n_mu0, meas_n_mu1]
        self.meas_n_std_dev[3, :] = [meas_n_std_dev0, meas_n_std_dev1]
        self.D[3] = np.diag([5, 5, meas_n_std_dev0, meas_n_std_dev1])  # diag([30;30;30;30])
        self.R[3] = np.dot(self.D[3], self.D[3].T)


if __name__ == '__main__':
    model_params = model()
    print(model_params)
