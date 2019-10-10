import cv2
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.pose_to_3d.converter import *
from utils.openpose.body import pose_keypoints
from utils.AIR import to_joint
from act2act.train import *
from act2act.draw import init_axis, draw_parts
from data.normalization import normalize_body_data


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


def pose_to_AIR(skel_3d_upper):
    body = [to_joint([0, 0, 0])] * 25
    body[ 8] = to_joint(skel_3d_upper[config.upper.index(config.RShoulder)])
    body[ 4] = to_joint(skel_3d_upper[config.upper.index(config.LShoulder)])
    body[ 9] = to_joint(skel_3d_upper[config.upper.index(config.RElbow)])
    body[ 5] = to_joint(skel_3d_upper[config.upper.index(config.LElbow)])
    body[10] = to_joint(skel_3d_upper[config.upper.index(config.RWrist)])
    body[ 6] = to_joint(skel_3d_upper[config.upper.index(config.LWrist)])

    body[ 0] = to_joint(skel_3d_upper[config.upper.index(config.MidHip)])
    body[20] = to_joint(skel_3d_upper[config.upper.index(config.Neck)])
    body[ 3] = to_joint(skel_3d_upper[config.upper.index(config.Nose)])
    return body


def webcam():
    # initialize web-cam
    image_width = 640
    image_height = 480
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
    print("Camera opened")

    # initialize figure
    fig = plt.figure(figsize=(30, 10))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    plt.ion()

    # initialize act2act model
    sess = create_session()
    FLAGS.load = 10000
    FLAGS.use_cpu = True
    model = create_model(sess)
    print("Model created")
    human = list()
    robot = list()

    # iterate
    while True:
        # get camera frame
        ret, frame = cap.read()

        # run OpenPose to get 2d pose
        # begin = time.time()
        key_points, output_data = pose_keypoints(frame)
        # end = time.time()
        # print(f'OpenPose: elapsed time is {end-begin}')

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
            skel_2d_trans = skel_2d.transpose()  # [(x,y) x n_joint]
            z_out_trans = z_out.reshape(-1)
            show_skeletons(ax1, ax2, skel_2d_trans, z_out_trans)
            plt.show(block=False)

            # generate robot pose
            z_out_expand = np.expand_dims(z_out_trans, axis=1)
            skel_3d_upper = np.append(skel_2d_upper, z_out_expand, axis=1)
            body = pose_to_AIR(skel_3d_upper)
            human.append(np.hstack([0.5, normalize_body_data(body, human_feature_type)]))
            robot.append([0.0] * 10 if len(robot) < source_len else next_seq[0][0])
            human = human[-source_len:]
            robot = robot[-source_len:]

            init_axis(ax3)
            if len(human) < source_len:
                continue
            context_inp = np.array([human])
            encoder_inp = np.array([robot[:-1]])
            decoder_inp = np.array([[robot[-1]] * target_len])
            decoder_out = decoder_inp
            _, next_seq, _ = model.step(sess, context_inp, encoder_inp, decoder_inp, decoder_out,
                                        forward_only=True, srnn_seeds=True)

            # draw robot pose
            angles = denormalize_feature(next_seq[0][0], robot_feature_type)
            pelvis, neck, head, lshoulder, lelbow, lwrist, rshoulder, relbow, rwrist = angles
            draw_parts(ax3, [pelvis, neck, head])
            draw_parts(ax3, [neck, lshoulder, lelbow, lwrist])
            draw_parts(ax3, [neck, rshoulder, relbow, rwrist])

        else:
            ax1.clear()
            ax2.clear()
            ax3.clear()

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()


if __name__ == "__main__":
    webcam()
