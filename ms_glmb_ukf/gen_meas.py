from gen_truth import truth
from gen_model import model
from target import gen_msobservation_fn
import numpy as np


def gen_observation_fn(model, s, X, W):
    # linear observation equation (position components only)
    WW = np.zeros((model.D[s].shape[0], X.shape[1]))  # noiseless
    if W is 'noise':
        vec_D = np.diag(model.D[s])
        WW[0:4, :] = np.dot(model.D[s],  np.random.randn(vec_D.shape[0], X.shape[1]))
        WW[2:4, :] = WW[2:4, :] + np.array([[model.meas_n_mu[s, 0]], [model.meas_n_mu[s, 1]]])
    Z = gen_msobservation_fn(model, X, WW, s)  # Z = HX
    return Z


class meas:
    def __init__(self, model, truth):
        # variables
        self.K = truth.K
        self.Z = {}
        for k in range(0, truth.K):
            for s in range(model.N_sensors):
                self.Z[(k, s)] = np.empty((model.z_dim[s], 0))
        #
        # generate measurements
        for k in range(0, truth.K):
            for s in range(model.N_sensors):
                if truth.N[k] > 0:
                    idx = np.nonzero(np.random.rand(truth.N[k], 1) <= model.P_D[s])[0]  # detected target indices
                    # single target observations if detected
                    self.Z[k, s] = gen_observation_fn(model, s, truth.X[k][:, idx], 'noise')

                N_c = np.random.poisson(model.lambda_c[s, 0])  # number of clutter points
                range_c_log_extent = np.copy(model.range_c[s])
                range_c_log_extent[[2, 3], :] = np.log(range_c_log_extent[[2, 3], :])  # log extent
                C = np.tile(range_c_log_extent[:, 0].reshape(-1, 1), (1, N_c))
                C = C + np.diagflat(np.dot(range_c_log_extent, np.array([[-1], [1]]))) @ \
                    np.random.rand(model.z_dim[s], N_c)  # clutter generation
                self.Z[k, s] = np.column_stack((self.Z[k, s], C))  # measurement is union of detections and clutter
            #
        #


if __name__ == '__main__':
    model_params = model()
    truth_params = truth(model_params)
    meas_params = meas(model_params, truth_params)
    print(meas_params)
