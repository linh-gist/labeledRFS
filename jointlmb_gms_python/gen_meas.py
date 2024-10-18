from gen_truth import truth
from gen_model import model
import numpy as np


def gen_observation_fn(model, X, W):
    # linear observation equation (position components only)
    if W is 'noise':
        W = np.dot(model.D, np.random.randn(model.D.shape[1], X.shape[1]));
    if W is 'noiseless':
        W = np.zeros(model.D.shape[0]);
    if len(X) == 0:
        Z = [];
    else:
        Z = np.dot(model.H, X) + W;
    return Z


class meas:
    def __init__(self, model, truth):
        # variables
        self.K = truth.K;
        self.Z = {key: np.empty((model.z_dim, 0)) for key in range(0, truth.K)};

        # generate measurements
        for k in range(0, truth.K):
            if truth.N[k] > 0:
                idx = np.nonzero(np.random.rand(truth.N[k], 1) <= model.P_D)[0];  # detected target indices
                self.Z[k] = gen_observation_fn(model, truth.X[k][:, idx],
                                               'noise');  # single target observations if detected

            N_c = np.random.poisson(model.lambda_c);  # number of clutter points
            C = np.tile(model.range_c[:, 0].reshape(-1, 1), (1, N_c)) + \
                np.diagflat(np.dot(model.range_c, np.array([[-1], [1]]))) @ \
                np.random.rand(model.z_dim, N_c);  # clutter generation
            self.Z[k] = np.column_stack((self.Z[k], C));  # measurement is union of detections and clutter


if __name__ == '__main__':
    model_params = model()
    truth_params = truth(model_params)
    meas_params = meas(model_params, truth_params)
    print(meas_params)
