import glob
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data.constants import *
from data.extract_data import b_iter
from data.normalization import denormalize_feature


def main():
    # show all test data
    n_data, data_files, data_names = get_data_files(dst_test)
    print('There are %d data.' % n_data)
    for action_index in range(n_data):
        print('%d: %s' % (action_index, data_names[action_index]))

    # select data name to draw
    while True:
        var = int(input("Input data number to display: "))
        files = [name for name in data_files if data_names[var] in name]

        human_joint_positions = list()
        robot_joint_positions = list()
        if not b_iter:
            with np.load(files[0]) as data:
                human_seq = data['human_seq']
                robot_seq = data['robot_seq']
                for line in human_seq:
                    human_joint_positions.append(denormalize_feature(line[dist_len:], human_feature_type))
                for line in robot_seq:
                    robot_joint_positions.append(denormalize_feature(line, robot_feature_type))
        for file in files:
            with np.load(file) as data:
                human_seq = data['human_seq']
                robot_seq = data['robot_seq']
                human_joint_positions.append(denormalize_feature(human_seq[-1][dist_len:], human_feature_type))
                robot_joint_positions.append(denormalize_feature(robot_seq[source_len], robot_feature_type))

        draw([human_joint_positions, robot_joint_positions], save_path=None, b_show=True)


def get_data_files(path):
    data_files = glob.glob(os.path.normpath(os.path.join('../data', path, "*.npz")))
    data_files.sort()

    data_names = list()
    n_data = 0
    for data_file in data_files:
        data_name = os.path.splitext(os.path.basename(data_file))[0][:-4]
        if data_name not in data_names:
            data_names.append(data_name)
            n_data += 1

    return n_data, data_files, data_names


def draw(angles, save_path=None, b_show=False):
    fig = plt.figure()
    axes = [fig.add_subplot(1, len(angles), idx + 1, projection='3d') for idx in range(len(angles))]
    anim = animation.FuncAnimation(fig, animate_3d, interval=100, blit=True, fargs=(angles, axes),
                                   frames=len(angles[0]), repeat=True)
    writer = animation.writers['ffmpeg'](fps=10)
    anim.save(save_path, writer=writer, dpi=250) if save_path else None
    plt.show() if b_show else None
    plt.close()


def init_axis(ax):
    ax.clear()

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    max = 2.
    ax.set_xlim3d(-max, max)
    ax.set_ylim3d(0, 2 * max)
    ax.set_zlim3d(-max, max)

    ax.view_init(elev=-80, azim=90)


def animate_3d(f, angles, axes):
    ret_artists = list()
    for idx in range(len(angles)):
        init_axis(axes[idx])
        cur_angles = angles[idx][f] if f < len(angles[idx]) else angles[idx][-1]
        pelvis, neck, head, lshoulder, lelbow, lwrist, rshoulder, relbow, rwrist = cur_angles
        ret_artists.extend(draw_parts(axes[idx], [pelvis, neck, head]))
        ret_artists.extend(draw_parts(axes[idx], [neck, lshoulder, lelbow, lwrist]))
        ret_artists.extend(draw_parts(axes[idx], [neck, rshoulder, relbow, rwrist]))
        ret_artists.extend([axes[idx].text(0, 0, 0, '{0}/{1}'.format(f + 1, len(angles[idx])))])
    return ret_artists


def draw_parts(ax, joints):
    def add_points(points):
        xs, ys, zs = list(), list(), list()
        for point in points:
            xs.append(point[0])
            ys.append(point[1])
            zs.append(point[2])
        return xs, ys, zs

    xs, ys, zs = add_points(joints)
    ret = ax.plot(xs, ys, zs, color='b')
    return ret


if __name__ == "__main__":
    main()
