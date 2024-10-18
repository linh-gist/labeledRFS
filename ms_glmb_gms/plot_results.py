import numpy as np
import matplotlib.pyplot as plt

def plot_results(model, truth, meas, est):
    X_track, k_birth, k_death = extract_tracks(truth.X, truth.track_list, truth.total_tracks)

    labelcount = countestlabels(meas, est)
    ca_obj = ca()
    colorarray = ca_obj.makecolorarray(labelcount)
    est.total_tracks = labelcount
    est.track_list = {key: np.empty(0, dtype=int) for key in range(0, truth.K)};
    for k in range(0, truth.K):
        for eidx in range(0, est.X[k].shape[1]):
            est.track_list[k] = np.append(est.track_list[k], assigncolor(est.L[k][:, eidx], colorarray))

    Y_track, l_birth, l_death = extract_tracks(est.X, est.track_list, est.total_tracks)

    # plot ground truths
    limit = np.array([model.range_c[0, 0, 0], model.range_c[0, 0, 1], model.range_c[1, 0, 0], model.range_c[1, 0, 1],
                      model.range_c[2, 0, 0], model.range_c[2, 0, 1]])
    plt.figure(figsize=(9, 3))
    ax = plt.axes(projection='3d')
    for i in range(0, truth.total_tracks):
        Pt = X_track[:, np.arange(k_birth[i], k_death[i], 1), i]
        Pt = Pt[[0, 2, 4], :]
        plt.plot(Pt[0, :], Pt[1, :], Pt[2, :], 'k-')
        plt.plot(Pt[0, 0], Pt[1, 0], Pt[2, 0], 'ko', 6)
        plt.plot(Pt[0, (k_death[i] - k_birth[i] - 1)], Pt[1, (k_death[i] - k_birth[i] - 1)],
                  Pt[2, (k_death[i] - k_birth[i] - 1)], 'k^', 6)
    ax.set_xlim3d(limit[0], limit[1])
    ax.set_ylim3d(limit[0], limit[1])
    ax.set_zlim3d(limit[0], limit[1])
    plt.title('Ground Truths')
    # plt.show()

    for s in range(model.N_sensors):
        # plot tracks and measurements in x/y
        # plot x measurement
        fig, (axs1, axs2, axs3) = plt.subplots(3)
        for k in range(0, meas.K):
            if meas.Z[(k, s)].size == 0:
                continue
            plt_x_meas = axs1.scatter(k * np.ones((meas.Z[(k, s)].shape[1], 1)), meas.Z[(k, s)][0, :], marker='x', s=50,
                                      color=0.7 * np.ones((1, 3)), label='Measurements')
            plt_y_meas = axs2.scatter(k * np.ones((meas.Z[(k, s)].shape[1], 1)), meas.Z[(k, s)][1, :], marker='x', s=50,
                                      color=0.7 * np.ones((1, 3)), label='Measurements')
            plt_z_meas = axs3.scatter(k * np.ones((meas.Z[(k, s)].shape[1], 1)), meas.Z[(k, s)][2, :], marker='x', s=50,
                                      color=0.7 * np.ones((1, 3)), label='Measurements')
        # plot x, y, z track
        for i in range(0, truth.total_tracks):
            P = X_track[:, np.arange(k_birth[i], k_death[i], 1), i]
            P = P[[0, 2, 4], :]
            plt_x_truth = axs1.plot(np.arange(k_birth[i], k_death[i], 1), P[0, :], linestyle='-', linewidth=1,
                                    color=0 * np.ones((1, 3)), label='True tracks')
            plt_y_truth = axs2.plot(np.arange(k_birth[i], k_death[i], 1), P[1, :], linestyle='-', linewidth=1,
                                    color=0 * np.ones((1, 3)), label='True tracks')
            plt_z_truth = axs3.plot(np.arange(k_birth[i], k_death[i], 1), P[2, :], linestyle='-', linewidth=1,
                                    color=0 * np.ones((1, 3)), label='True tracks')
        # plt.show()
        # plot x, y, z estimate
        for k in range(meas.K):
            if len(est.X[k]) == 0:
                continue
            P = est.X[k][[0, 2, 4]]
            L = est.L[k]
            for eidx in range(P.shape[1]):
                color = assigncolor(L[:, eidx], colorarray)[0]
                plt_x_est = axs1.plot(k, P[0, eidx], marker='.', color=colorarray.rgb[:, color], label='Estimates')
                plt_y_est = axs2.plot(k, P[1, eidx], marker='.', color=colorarray.rgb[:, color], label='Estimates')
                plt_z_est = axs3.plot(k, P[2, eidx], marker='.', color=colorarray.rgb[:, color], label='Estimates')
        axs1.legend(handles=[plt_x_meas, plt_x_truth[0], plt_x_est[0]])
        axs1.set_xlabel('Time')
        axs1.set_ylabel('x-coordinate (m)')
        axs2.set_xlabel('Time');
        axs2.set_ylabel('y-coordinate (m)')
        axs3.set_xlabel('Time');
        axs3.set_ylabel('z-coordinate (m)')
        plt.show()


def plot_truth_meas(model, truth, meas):
    X_track, k_birth, k_death = extract_tracks(truth.X, truth.track_list, truth.total_tracks)

    # plot ground truths
    limit = np.array([model.range_c[0, 0], model.range_c[0, 1], model.range_c[1, 0], model.range_c[1, 1]])
    plt.figure(figsize=(9, 3))
    for i in range(0, truth.total_tracks):
        Pt = X_track[:, np.arange(k_birth[i], k_death[i], 1), i]
        Pt = Pt[[0, 2], :]
        plt.plot(Pt[0, :], Pt[1, :], 'k-');
        plt.plot(Pt[0, 0], Pt[1, 0], 'ko', 6);
        plt.plot(Pt[0, (k_death[i] - k_birth[i] - 1)], Pt[1, (k_death[i] - k_birth[i] - 1)], 'k^', 6)
    plt.axis(limit)
    plt.title('Ground Truths')
    # plt.show()

    # plot tracks and measurements in x/y
    # plot x measurement
    fig, (axs1, axs2) = plt.subplots(2, figsize=(12, 12))
    for k in range(0, meas.K):
        if meas.Z[k].size is not 0:
            plt_x_meas = axs1.scatter(k * np.ones((meas.Z[k].shape[1], 1)), meas.Z[k][0, :], marker='x',
                     s=50, color=0.7 * np.ones((1, 3)), label='Measurements')
    # plot x track
    for i in range(0, truth.total_tracks):
        Px = X_track[:, np.arange(k_birth[i], k_death[i], 1), i]
        Px = Px[[0, 2], :]
        plt_x_truth = axs1.plot(np.arange(k_birth[i], k_death[i], 1), Px[0, :], linestyle='-', linewidth=1,
                 color=0 * np.ones(3), label='True tracks')
    axs1.legend(handles=[plt_x_meas, plt_x_truth[0]])
    axs1.set_xlabel('Time')
    axs1.set_ylabel('x-coordinate (m)')

    # plot y measurement
    # plt.subplot(212)
    for k in range(0, meas.K):
        if meas.Z[k].size is not 0:
            plt_y_meas = axs2.scatter(k * np.ones((meas.Z[k].shape[1], 1)), meas.Z[k][1, :], marker='x',
                     s=50, color=0.7 * np.ones((1, 3)), label='Measurements')
    # plot y track
    for i in range(0, truth.total_tracks):
        Py = X_track[:, np.arange(k_birth[i], k_death[i], 1), i]
        Py = Py[[0, 2], :]
        plt_y_truth = axs2.plot(np.arange(k_birth[i], k_death[i], 1), Py[1, :], linestyle='-', linewidth=1,
                 color=0 * np.ones(3), label='True tracks')
    # plt.legend('Estimates')
    axs2.set_xlabel('Time');
    axs2.set_ylabel('y-coordinate (m)')
    axs2.legend(handles=[plt_y_meas, plt_y_truth[0]])
    # plt.show()

    return fig, axs1, axs2

class ca:
    def makecolorarray(self, nlabels):
        lower = 0.1
        upper = 0.9
        rrr = np.random.rand(1, nlabels) * (upper - lower) + lower
        ggg = np.random.rand(1, nlabels) * (upper - lower) + lower
        bbb = np.random.rand(1, nlabels) * (upper - lower) + lower
        self.rgb = np.concatenate((rrr, ggg, bbb))
        self.lab = np.empty((nlabels), dtype=object)
        self.cnt = -1

        return self


def assigncolor(label, colorarray):
    str = np.array2string(label, separator='*')[1:-1] + '*'
    tmp = (str == colorarray.lab)
    if np.nonzero(tmp)[0].size > 0:
        idx = np.nonzero(tmp)[0]
    else:
        colorarray.cnt = colorarray.cnt + 1;
        colorarray.lab[colorarray.cnt] = str
        idx = colorarray.cnt

    return idx


def countestlabels(meas, est):
    labelstack = np.empty((2, 0), dtype=int)
    for k in range(0, meas.K):
        labelstack = np.concatenate((labelstack, est.L[k]), axis=1)
    c, _, _ = np.unique(labelstack, return_index=True, return_inverse=True, axis=1)
    count = c.shape[1]
    return count


def extract_tracks(X, track_list, total_tracks):
    K = len(X)
    k = K - 1
    x_dim = X[k].shape[0]
    while x_dim == 0:
        x_dim = X[k].shape[0]
        k = k - 1
    X_track = np.zeros((x_dim, K, total_tracks))
    X_track[:] = np.NaN
    k_birth = np.zeros((total_tracks, 1), dtype=int)
    k_death = np.zeros((total_tracks, 1), dtype=int)

    max_idx = 0;
    for k in range(0, K):
        if X[k].size is not 0:
            X_track[:, k, track_list[k]] = X[k]
        else:
            continue
        if max(track_list[k]) > max_idx:  # new target born?
            idx = np.nonzero(track_list[k] > max_idx)[0]
            k_birth[track_list[k][idx]] = k

        max_idx = max(track_list[k])
        k_death[track_list[k]] = k + 1

    return X_track, k_birth, k_death


def get_comps(X, c):
    if len(X) == 0:
        Xc = {};
    else:
        Xc = X[c, :]
    return Xc
