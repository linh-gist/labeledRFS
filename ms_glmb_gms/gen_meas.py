from gen_truth import truth
from gen_model import model
import numpy as np


def gen_observation_fn(model, s, X, W):
    # linear observation equation (position components only)
    if W is 'noise':
        W = np.dot(model.D[s], np.random.randn(model.D[s].shape[1], X.shape[1]))
    if W is 'noiseless':
        W = np.zeros(model.D[s].shape[0])
    if len(X) == 0:
        Z = []
    else:
        Z = np.dot(model.H[s], X) + W
    return Z


class meas:
    def __init__(self, model, truth):
        # variables
        self.K = truth.K
        self.Z = {}
        for k in range(0, truth.K):
            for s in range(model.N_sensors):
                self.Z[(k, s)] = np.empty((model.z_dim[s, 0], 0))
        #
        # generate measurements
        for k in range(0, truth.K):
            for s in range(model.N_sensors):
                if truth.N[k] > 0:
                    idx = np.nonzero(np.random.rand(truth.N[k], 1) <= model.P_D[s])[0]  # detected target indices
                    self.Z[k, s] = gen_observation_fn(model, s, truth.X[k][:, idx],
                                                   'noise')  # single target observations if detected

                N_c = np.random.poisson(model.lambda_c[s, 0])  # number of clutter points
                C = np.tile(model.range_c[s][:, 0].reshape(-1, 1), (1, N_c)) + \
                    np.diagflat(np.dot(model.range_c[s], np.array([[-1], [1]]))) @ \
                    np.random.rand(model.z_dim[s, 0], N_c)  # clutter generation
                self.Z[k, s] = np.column_stack((self.Z[k, s], C))  # measurement is union of detections and clutter
            #
        #


if __name__ == '__main__':
    model_params = model()
    truth_params = truth(model_params)
    meas_params = meas(model_params, truth_params)
    print(meas_params)
