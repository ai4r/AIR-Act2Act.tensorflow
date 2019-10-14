import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.pose_to_3d.converter import *
from utils.openpose.body import pose_keypoints


def show_skeletons(ax_2d, ax_3d, skel_2d, z_out, z_gt=None):
    # init figures and variable
    ax_2d.clear()
    ax_3d.clear()
    edges = np.array([[config.MidHip, config.Neck],
                      [config.Neck, config.Nose],
                      [config.Neck, config.LShoulder],
                      [config.LShoulder, config.LElbow],
                      [config.LElbow, config.LWrist],
                      [config.Neck, config.RShoulder],
                      [config.RShoulder, config.RElbow],
                      [config.RElbow, config.RWrist]])

    # draw 3d
    for edge in edges:
        ax_3d.plot(skel_2d[0, edge], z_out[edge], skel_2d[1, edge], color='r')
        if z_gt is not None:
            ax_3d.plot(skel_2d[0, edge], z_gt[edge], skel_2d[1, edge], color='g')

    ax_3d.set_aspect('equal')
    ax_3d.set_xlabel("x"), ax_3d.set_ylabel("z"), ax_3d.set_zlabel("y")
    ax_3d.set_xlim3d([-2, 2]), ax_3d.set_ylim3d([2, -2]), ax_3d.set_zlim3d([2, -2])
    ax_3d.view_init(elev=10, azim=-45)

    # draw 2d
    for edge in edges:
        ax_2d.plot(skel_2d[0, edge], skel_2d[1, edge], color='r')

    ax_2d.set_aspect('equal')
    ax_2d.set_xlabel("x"), ax_2d.set_ylabel("y")
    ax_2d.set_xlim([-2, 2]), ax_2d.set_ylim([2, -2])


def webcam():
    # initialize web-cam
    image_width = 640
    image_height = 480
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

    # initialize figure
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plt.ion()

    # iterate
    while True:
        # get camera frame
        ret, frame = cap.read()

        # run OpenPose to get 2d pose
        key_points, output_data = pose_keypoints(frame)
        cv2.imshow(f'{image_width}x{image_height}', output_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # continue if no people is detected
        if len(key_points.shape) != 3:
            continue

        # else, convert 2d pose to 3d pose
        user_key_points = key_points[0]
        if user_key_points[config.RShoulder].any() and user_key_points[config.LShoulder].any():
            # normalize and refine 2d pose
            skel_2d = normalize_pose(user_key_points)
            skel_2d = refine_pose(skel_2d)

            # infer z component
            skel_2d_upper = skel_2d[config.upper, :]
            v_input = pose_to_input(skel_2d_upper)
            z_out = infer_z(v_input)
            z_out = z_out.detach().cpu().numpy()

            # draw 3d skeleton
            skel_2d = skel_2d.transpose()  # [(x,y) x n_joint]
            z_out = z_out.reshape(-1)
            show_skeletons(ax1, ax2, skel_2d, z_out)
            plt.show(block=False)

        else:
            ax1.clear()
            ax2.clear()

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()


if __name__ == "__main__":
    webcam()
