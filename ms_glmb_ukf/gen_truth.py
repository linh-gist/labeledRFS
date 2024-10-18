import numpy as np
from gen_model import model


def gen_newstate_fn(model, Xd, V):
    # linear state space equation (CV model)
    if V is 'noise':
        V = np.dot(model.B, np.randn(model.B.shape[1], len(Xd)))
    if V is 'noiseless':
        V = np.zeros((model.B.shape[0]))

    if len(Xd) == 0:
        X = []
    else:
        X = np.dot(model.F, Xd) + V
    return X


class truth:
    def __init__(self, model):
        # variables
        self.K = 100  # length of data/number of scans
        self.X = {key: np.empty((model.x_dim, 0)) for key in range(0, self.K)}  # ground truth for states of targets
        self.N = {key: 0 for key in range(0, self.K)}  # ground truth for number of targets
        self.L = np.zeros((self.K, 1))  # ground truth for labels of targets (k,i)
        self.track_list = {key: np.empty(0, dtype=int) for key in
                           range(0, self.K)}  # absolute index target identities (plotting)
        self.total_tracks = 0  # total number of appearing tracks

        # target initial states and birth/death times
        nbirths = 12

        tbirth = np.zeros((nbirths), np.int)
        tdeath = np.zeros((nbirths), np.int)
        xstart = np.zeros((model.x_dim, nbirths))

        xstart[:, 0:1] = np.array([[2.52], [0.03], [0.71], [0.01], [0.825], [0], [np.log(0.3)], [np.log(0.3)], [np.log(0.84)]])
        tbirth[0] = 1
        tdeath[0] = 70
        xstart[:, 1:2] = np.array([[2.5], [0.01], [2.2], [-0.02], [0.825], [0], [np.log(0.3)], [np.log(0.3)], [np.log(0.84)]])
        tbirth[1] = 1
        tdeath[1] = self.K + 1
        xstart[:, 2:3] = np.array([[5.5], [-0.01], [2.2], [-0.02], [0.825], [0], [np.log(0.3)], [np.log(0.3)], [np.log(0.84)]])
        tbirth[2] = 1
        tdeath[2] = 70

        xstart[:, 3:4] = np.array([[5.5], [-0.03], [0.71], [0.024], [0.825], [0], [np.log(0.3)], [np.log(0.3)], [np.log(0.84)]])
        tbirth[3] = 20
        tdeath[3] = self.K + 1
        xstart[:, 4:5] = np.array([[2.5], [0.035], [2.2], [-0.01], [0.825], [0], [np.log(0.3)], [np.log(0.3)], [np.log(0.84)]])
        tbirth[4] = 20
        tdeath[4] = self.K + 1
        xstart[:, 5:6] = np.array([[2.52], [0.024], [0.71], [0.024], [0.825], [0], [np.log(0.3)], [np.log(0.3)], [np.log(0.84)]])
        tbirth[5] = 20
        tdeath[5] = self.K + 1

        xstart[:, 6:7] = np.array([[5.5], [-0.03], [2.2], [0.01], [0.825], [0], [np.log(0.3)], [np.log(0.3)], [np.log(0.84)]])
        tbirth[6] = 40
        tdeath[6] = self.K + 1
        xstart[:, 7:8] =np.array([[5.5], [-0.03], [0.71], [-0.005], [0.825], [0], [np.log(0.3)], [np.log(0.3)], [np.log(0.84)]])
        tbirth[7] = 40
        tdeath[7] = self.K + 1

        xstart[:, 8:9] = np.array([[2.5], [0.04], [2.2], [-0.03], [0.825], [0], [np.log(0.3)], [np.log(0.3)], [np.log(0.84)]])
        tbirth[8] = 60
        tdeath[8] = self.K + 1
        xstart[:, 9:10] = np.array([[2.5], [-0.01], [0.71], [0.04], [0.825], [0], [np.log(0.3)], [np.log(0.3)], [np.log(0.84)]])
        tbirth[9] = 60
        tdeath[9] = self.K + 1

        xstart[:, 10:11] = np.array([[5.5], [0.025], [0.71], [0.04], [0.825], [0], [np.log(0.3)], [np.log(0.3)], [np.log(0.84)]])
        tbirth[10] = 80
        tdeath[10] = self.K + 1
        xstart[:, 11:12] = np.array([[5.5], [0.005], [2.20], [0.04], [0.825], [0], [np.log(0.3)], [np.log(0.3)], [np.log(0.84)]])
        tbirth[11] = 80
        tdeath[11] = self.K + 1

        # generate the tracks
        for targetnum in range(nbirths):
            targetstate = xstart[:, targetnum]
            for k in range(tbirth[targetnum] - 1, min(tdeath[targetnum], self.K)):
                targetstate = gen_newstate_fn(model, targetstate, 'noiseless')
                self.X[k] = np.column_stack((self.X[k], targetstate))
                self.track_list[k] = np.append(self.track_list[k], targetnum)
                self.N[k] = self.N[k] + 1
        self.total_tracks = nbirths


if __name__ == '__main__':
    model_params = model()
    truth_params = truth(model_params)
    print(truth_params)
