import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_results(model, truth, meas, est):
    X_track, k_birth, k_death = extract_tracks(truth.X, truth.track_list, truth.total_tracks)
    k_birth, k_death = k_birth.flatten(), k_death.flatten()
    for i in range(X_track.shape[2]):
        X_track[[6, 7, 8], k_birth[i]: k_death[i], i] = np.exp(X_track[[6, 7, 8], k_birth[i]: k_death[i], i])
    labelcount = countestlabels(meas, est)
    ca_obj = ca()
    colorarray = ca_obj.makecolorarray(labelcount)
    est.total_tracks = labelcount
    est.track_list = {key: np.empty(0, dtype=int) for key in range(0, truth.K)};
    for k in range(0, truth.K):
        for eidx in range(0, est.X[k].shape[1]):
            est.track_list[k] = np.append(est.track_list[k], assigncolor(est.L[k][:, eidx], colorarray))

    Y_track, l_birth, l_death = extract_tracks(est.X, est.track_list, est.total_tracks)
    l_birth, l_death = l_birth.flatten(), l_death.flatten()
    for i in range(Y_track.shape[2]):
        Y_track[[6, 7, 8], l_birth[i]: l_death[i], i] = np.exp(Y_track[[6, 7, 8], l_birth[i]: l_death[i], i])

    # plot ground truths
    plt.figure(figsize=(9, 3))
    ax = plt.axes(projection='3d')
    for i in range(0, truth.total_tracks):
        Pt = X_track[:, np.arange(k_birth[i], k_death[i], 1), i]
        Pt = Pt[[0, 2, 4], :]
        plt.plot(Pt[0, :], Pt[1, :], Pt[2, :], 'k-')
        plt.plot(Pt[0, 0], Pt[1, 0], Pt[2, 0], 'ko', 6)
        plt.plot(Pt[0, (k_death[i] - k_birth[i] - 1)], Pt[1, (k_death[i] - k_birth[i] - 1)],
                  Pt[2, (k_death[i] - k_birth[i] - 1)], 'k^', 6)
    ax.set_xlim3d(model.XMAX[0], model.XMAX[1])
    ax.set_ylim3d(model.YMAX[0], model.YMAX[1])
    ax.set_zlim3d(model.ZMAX[0], model.ZMAX[1])
    plt.title('Ground Truths')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.show()

    for s in range(model.N_sensors):
        # plot tracks and measurements in x/y
        # plot x measurement
        fig, (axsX, axsY, axsZ) = plt.subplots(3)
        # plot x, y, z track
        for i in range(0, truth.total_tracks):
            P = X_track[:, np.arange(k_birth[i], k_death[i], 1), i]
            P = P[[0, 2, 4], :]
            plt_x_truth = axsX.plot(np.arange(k_birth[i], k_death[i], 1), P[0, :], linestyle='-', linewidth=1,
                                    color=0 * np.ones((1, 3)), label='True tracks')
            plt_y_truth = axsY.plot(np.arange(k_birth[i], k_death[i], 1), P[1, :], linestyle='-', linewidth=1,
                                    color=0 * np.ones((1, 3)), label='True tracks')
            plt_z_truth = axsZ.plot(np.arange(k_birth[i], k_death[i], 1), P[2, :], linestyle='-', linewidth=1,
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
                plt_x_est = axsX.plot(k, P[0, eidx], marker='.', color=colorarray.rgb[:, color], label='Estimates')
                plt_y_est = axsY.plot(k, P[1, eidx], marker='.', color=colorarray.rgb[:, color], label='Estimates')
                plt_z_est = axsZ.plot(k, P[2, eidx], marker='.', color=colorarray.rgb[:, color], label='Estimates')
        axsX.legend(handles=[plt_x_truth[0], plt_x_est[0]])
        axsX.set_xlabel('Time')
        axsX.set_ylabel('x-coordinate (m)')
        axsX.set_ylim(model.XMAX[0], model.XMAX[1])
        axsY.set_xlabel('Time')
        axsY.set_ylabel('y-coordinate (m)')
        axsY.set_ylim(model.YMAX[0], model.YMAX[1])
        axsZ.set_xlabel('Time')
        axsZ.set_ylabel('z-coordinate (m)')
        axsZ.set_ylim(model.ZMAX[0], model.ZMAX[1])
        plt.show()

    plot_video(model, est, colorarray)


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


def plot_video(model, est, colorarray):
    size = (1066, 600)
    out = cv2.VideoWriter('./ellipsoid_plot.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, size)
    print("Saving tracking result in video 'ellipsoid_plot.mp4'...")
    fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    fig.set_size_inches(size[0] / fig.dpi, size[1] / fig.dpi)
    ax = fig.add_subplot(111, projection='3d')
    offset = 0.5
    plt.tight_layout(pad=0)
    for k in range(len(est.X)):
        print("Saved Frame", k)
        targets = est.X[k]
        ax.plot([model.XMAX[0], model.XMAX[1]], [0, 0])
        ax.plot([model.XMAX[0], model.XMAX[0]], [model.YMAX[0], model.YMAX[1]])
        ax.plot([model.XMAX[1], model.XMAX[1]], [model.YMAX[0], model.YMAX[1]])
        ax.plot([model.XMAX[0], model.XMAX[1]], [model.YMAX[1], model.YMAX[1]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim3d(model.XMAX[0] - offset, model.XMAX[1] + offset)
        ax.set_ylim3d(model.YMAX[0] - offset, model.YMAX[1] + offset)
        ax.set_zlim3d(model.ZMAX[0], model.ZMAX[1])
        for eidx, tt in enumerate(targets.T):
            ellipsoid = tt[[0, 2, 4, 6, 7, 8]]
            ellipsoid[3:6] = np.exp(ellipsoid[3:6])
            index = assigncolor(est.L[k][:, eidx], colorarray)
            color = colorarray.rgb[:, index]
            label = str(np.array2string(est.L[k][:, eidx], separator="."))[1:-1]
            # Plotting an ellipsoid
            cx, cy, cz, rx, ry, rz = ellipsoid
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = rx * np.outer(np.cos(u), np.sin(v)) + cx
            y = ry * np.outer(np.sin(u), np.sin(v)) + cy
            z = rz * np.outer(np.ones_like(u), np.cos(v)) + cz
            ax.plot_wireframe(x, y, z, rstride=10, cstride=10, color=color.T)  # input color
            ax.text(cx, cy, cz + rz, label, size=12, color='green')
        # redraw the canvas
        fig.canvas.draw()
        # convert canvas to image
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        str_show = 'Frame {}'.format(k)
        img = cv2.putText(img, str_show, org=(img.shape[1] - 150, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                          fontScale=0.8, color=(255, 0, 0), thickness=2)
        cv2.imshow("Visual", img)
        cv2.waitKey(1)
        out.write(img)
        # plt.show()
        ax.cla()
    plt.close(fig)
    del fig
    out.release()
