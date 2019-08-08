import glob
import os
import numpy as np
from data.extract_data import to_angle
from utils.nao import solve_kinematics

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


def main():
    # show all test data
    n_data, data_files, data_names = get_data_files("../data/test")
    print('There are %d data.' % n_data)
    for action_index in range(n_data):
        print('%d: %s' % (action_index, data_names[action_index]))

    # select data name to draw
    var = int(input("Input data number to display: "))
    test_files = [name for name in data_files if data_names[var] in name]

    robot_angle_sequence = list()
    for test_file in test_files:
        array = np.loadtxt(test_file, dtype='float32')
        robot_features = array[-1][1:]
        robot_angles = to_angle(robot_features)
        robot_angle_sequence.append(robot_angles)
    save_anim(robot_angle_sequence, 'test.mp4', show=True)


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


def save_anim(robot_angle_sequence, anim_path, show=False):
    # draw skeleton and save animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    anim = animation.FuncAnimation(fig, animate_3d, interval=100, blit=True, fargs=(robot_angle_sequence, ax),
                                   frames=len(robot_angle_sequence), repeat=False)
    writer = animation.writers['ffmpeg'](fps=10)
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


def animate_3d(f, robot_angle_sequence, ax):
    init_axis(ax)

    robot_angles = robot_angle_sequence[f]
    pelvis, neck, head, lshoulder, lelbow, lwrist, rshoulder, relbow, rwrist = solve_kinematics(*robot_angles)

    xs, ys, zs = add_points([pelvis, neck, head])
    lobj1 = ax.plot(xs, ys, zs, color='b')
    xs, ys, zs = add_points([neck, lshoulder, lelbow, lwrist])
    lobj2 = ax.plot(xs, ys, zs, color='b')
    xs, ys, zs = add_points([neck, rshoulder, relbow, rwrist])
    lobj3 = ax.plot(xs, ys, zs, color='b')

    lines = lobj1 + lobj2 + lobj3
    texts = [ax.text(0, 0, 0, '{0}/{1}'.format(f + 1, len(robot_angle_sequence)))]

    return tuple(lines) + tuple(texts)


def add_points(points):
    xs, ys, zs = list(), list(), list()
    for point in points:
        xs.append( point[2])
        ys.append(-point[1])
        zs.append( point[0])
    return xs, ys, zs


if __name__ == "__main__":
    main()
