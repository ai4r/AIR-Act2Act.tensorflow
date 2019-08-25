import glob
import os
import numpy as np
from data.extract_data import to_angle, b_iter
from utils.nao import solve_kinematics
from constants import source_seq_size

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def main():
    # show all test data
    path = "../data/extracted files/in_30/out_20/train"
    n_data, data_files, data_names = get_data_files(path)
    print('There are %d data.' % n_data)
    for action_index in range(n_data):
        print('%d: %s' % (action_index, data_names[action_index]))

    # select data name to draw
    while True:
        var = int(input("Input data number to display: "))
        test_files = [name for name in data_files if data_names[var] in name]

        human_angle_sequence = list()
        robot_angle_sequence = list()
        if not b_iter:
            array = np.loadtxt(test_files[0], dtype='float32')
            for line in array[:source_seq_size[0]]:
                human_angles = to_angle(line[1:])
                human_angle_sequence.append(human_angles)
            for line in array[source_seq_size[0]:source_seq_size[0] * 2]:
                robot_angles = to_angle(line[1:])
                robot_angle_sequence.append(robot_angles)
        for test_file in test_files:
            array = np.loadtxt(test_file, dtype='float32')

            human_features = array[source_seq_size[0] - 1][1:]
            human_angles = to_angle(human_features)
            human_angle_sequence.append(human_angles)

            robot_features = array[source_seq_size[0] * 2][1:]
            robot_angles = to_angle(robot_features)
            robot_angle_sequence.append(robot_angles)

        save_anim([human_angle_sequence, robot_angle_sequence], show=True)


def get_data_files(path):
    data_files = glob.glob(os.path.normpath(path + "/*.sequence"))
    data_files.sort()

    data_names = list()
    n_data = 0
    for data_file in data_files:
        data_name = os.path.splitext(os.path.basename(data_file))[0][:-4]
        if data_name not in data_names:
            data_names.append(data_name)
            n_data += 1

    return n_data, data_files, data_names


def save_anim(angles, anim_path=None, show=False):
    # draw skeleton and save animation
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    anim = animation.FuncAnimation(fig, animate_3d, interval=100, blit=True, fargs=(angles, ax1, ax2),
                                   frames=len(angles[1]), repeat=True)
    writer = animation.writers['ffmpeg'](fps=10)
    if anim_path:
        anim.save(anim_path, writer=writer, dpi=250)
    if show:
        plt.show()
    plt.close()


def init_axis(ax):
    ax.clear()

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(0, 4)

    ax.view_init(30, -30)


def animate_3d(f, angles, ax1, ax2):
    human_angle_sequence, robot_angle_sequence = angles

    ret_artists = []
    init_axis(ax1)
    init_axis(ax2)

    if f >= len(human_angle_sequence):
        human_angles = human_angle_sequence[-1]
    else:
        human_angles = human_angle_sequence[f]
    robot_angles = robot_angle_sequence[f]

    pelvis, neck, head, lshoulder, lelbow, lwrist, rshoulder, relbow, rwrist = solve_kinematics(*human_angles)
    xs, ys, zs = add_points([pelvis, neck, head])
    ret_artists.extend(ax1.plot(xs, ys, zs, color='b'))
    xs, ys, zs = add_points([neck, lshoulder, lelbow, lwrist])
    ret_artists.extend(ax1.plot(xs, ys, zs, color='b'))
    xs, ys, zs = add_points([neck, rshoulder, relbow, rwrist])
    ret_artists.extend(ax1.plot(xs, ys, zs, color='b'))

    pelvis, neck, head, lshoulder, lelbow, lwrist, rshoulder, relbow, rwrist = solve_kinematics(*robot_angles)
    xs, ys, zs = add_points([pelvis, neck, head])
    ret_artists.extend(ax2.plot(xs, ys, zs, color='b'))
    xs, ys, zs = add_points([neck, lshoulder, lelbow, lwrist])
    ret_artists.extend(ax2.plot(xs, ys, zs, color='b'))
    xs, ys, zs = add_points([neck, rshoulder, relbow, rwrist])
    ret_artists.extend(ax2.plot(xs, ys, zs, color='b'))

    ret_artists.extend([ax1.text(0, 0, 0, '{0}/{1}'.format(f + 1, len(human_angle_sequence)))])
    ret_artists.extend([ax2.text(0, 0, 0, '{0}/{1}'.format(f + 1, len(robot_angle_sequence)))])

    return ret_artists


def add_points(points):
    xs, ys, zs = list(), list(), list()
    for point in points:
        xs.append( point[2])
        ys.append(-point[1])
        zs.append( point[0])
    return xs, ys, zs


if __name__ == "__main__":
    main()
