import os
import math
import random
import glob
import numpy as np
from tqdm import tqdm
from utils.nao import convert_to_nao
from utils.AIR import vectorize, read_joint
from constants import actions, source_seq_size, target_seq_size

n_step = 3
b_iter = False
human_type = 'angle'
robot_type = 'angle'
max_distance = 5.
p_test = 0.1
p_data = 1.0

src = "./joint files"
dst_train = os.path.normpath(os.path.join('./extracted files',
    'in_{0}'.format(source_seq_size[0]), 'out_{0}'.format(target_seq_size[0]), 'train'))
dst_test = os.path.normpath(os.path.join('./extracted files',
    'in_{0}'.format(source_seq_size[0]), 'out_{0}'.format(target_seq_size[0]), 'test'))


def main():
    if not os.path.exists(dst_train):
        os.makedirs(dst_train)
    if not os.path.exists(dst_test):
        os.makedirs(dst_test)

    for action in actions:
        print(action)
        human_files = glob.glob(src + '/' + action + '/C001*.joint')
        random.shuffle(human_files)
        n_data = int(len(human_files) * p_data)

        pbar = tqdm(total=n_data)
        for human_file in human_files[:n_data]:
            robot_file = human_file.replace('C001', 'C002')
            third_file = human_file.replace('C001', 'C003')

            human_info = read_joint(human_file)
            robot_info = read_joint(robot_file)
            third_info = read_joint(third_file)

            b_test = True if random.random() < p_test else False
            dst = dst_test if b_test else dst_train
            pred_len = 1 if b_test else target_seq_size[0]

            seq_generator = generate_seq(human_info, robot_info, third_info, seq_len=source_seq_size[0], pred_len=pred_len,
                                         n_step=n_step, b_iter=b_iter, human_type=human_type, robot_type=robot_type)

            index = 0
            for human_seq, robot_seq in seq_generator:
                data_name = human_file.replace("\\", "/").split("/")[-1].split('.')[0]
                data_file = dst + "/{}_{:03d}.sequence".format(data_name, index)
                np.savetxt(data_file, np.array(human_seq + robot_seq, dtype='float32'))
                index += 1

            pbar.update(1)

        pbar.close()


def generate_seq(human, robot, third, seq_len, pred_len=1, n_step=1, b_iter=True, human_type='angle',
                 robot_type='angle'):
    """
    Generate the sequence of features to be used as training & test data.
    Args
        human: human body information extracted from "C001*.joint" file
        robot: robot body information extracted from "C002*.joint" file
        third: human & robot body information extracted from "C003*.joint" file
        seq_len: length of input sequence
        pred_len: length of output sequence
        b_iter: whether to hold the first frame for a while
        human_type: type of features of human body
        robot_type: type of features of robot body
    Yields
        human_seq: sequence of human body features (input)
        robot_seq: sequence of robot body features (input and output)
    """
    # extract features
    n_seq = min(len(human), len(robot), len(third))
    human_features = []
    robot_features = []
    third_features = []
    human_cam_pos, human_cam_dir = get_camera(human[0][1]["joints"])
    robot_cam_pos, robot_cam_dir = get_camera(robot[0][0]["joints"])

    for f in range(n_seq):
        if all(x != 0 for x in human_cam_dir):
            move_camera(human[f][1]["joints"], cam_pos=human_cam_pos, cam_dir=human_cam_dir)
        if all(x != 0 for x in robot_cam_dir):
            move_camera(robot[f][0]["joints"], cam_pos=robot_cam_pos, cam_dir=robot_cam_dir)

    for f in range(0, n_seq, n_step):
        itr = seq_len - 1 if f == 0 and b_iter else 1  # hold the first frame for a while
        for _ in range(itr):
            # human body features
            if human_type == 'angle':
                human_angles = np.array(convert_to_nao(human[f][1]["joints"]))
                human_features.append(to_feature(human_angles).tolist())
            else:
                # to be updated (3d point)
                print('human_type should be angle')

            # robot body features
            if robot_type == 'angle':
                robot_angles = np.array(convert_to_nao(robot[f][0]["joints"]))
                robot_features.append(to_feature(robot_angles).tolist())
            else:
                # to be updated (3d point)
                print('robot_type should be angle')

            # distance features
            n_body = sum(1 for b in third[f] if b is not None)
            if n_body != 2:
                print('third camera information is wrong.')
                continue
            human_pos = vectorize(third[f][1]["joints"][0])
            robot_pos = vectorize(third[f][0]["joints"][0])
            dist_third = max_distance if all(v == 0 for v in human_pos) else distance(human_pos, robot_pos)

            human_pos = vectorize(human[f][1]["joints"][0])
            robot_pos = np.array([0, 0, 0])
            dist_human = max_distance if all(v == 0 for v in human_pos) else distance(human_pos, robot_pos)

            dist = min(dist_third, dist_human)
            third_features.append([dist / max_distance])

    # extract sequences
    for start in range(len(human_features) - seq_len - (pred_len - 1)):
        if all(v == [1.0] for v in third_features[start:start + seq_len]):
            continue
        human_seq = [third_features[i] + human_features[i] for i in range(start, start + seq_len           )]
        robot_seq = [third_features[i] + robot_features[i] for i in range(start, start + seq_len + pred_len)]
        yield human_seq, robot_seq


def to_feature(array):
    return (array + math.pi) / (math.pi * 2)


def to_angle(array):
    return array * (math.pi * 2) - math.pi


def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 + (p0[2] - p1[2])**2)


def normalize(vector):
    norm = norm_2(vector)
    if norm == 0:
        norm = np.finfo(vector.dtype).eps
    return vector / norm


def norm_2(vector):
    return np.linalg.norm(vector, axis=0, ord=2)


def get_camera(body):
    r_16_kinect = vectorize(body[16])
    r_12_kinect = vectorize(body[12])
    r_20_kinect = vectorize(body[20])
    r_0_kinect = vectorize(body[0])

    # rotation matrix from kinect to human coordinates
    z = normalize(np.cross(r_16_kinect - r_12_kinect, r_16_kinect - r_20_kinect))
    cam_pos = r_0_kinect + z
    cam_dir = -z

    return cam_pos, cam_dir


def move_camera(body, cam_pos, cam_dir):
    # rotation factors
    dist = math.sqrt(cam_dir[0] ** 2 + cam_dir[1] ** 2 + cam_dir[2] ** 2)
    dist_y = math.sqrt(cam_dir[0] ** 2 + cam_dir[2] ** 2)
    cos_x = dist_y / dist
    sin_x = -cam_dir[1] / dist
    cos_y = cam_dir[2] / dist_y
    sin_y = cam_dir[0] / dist_y

    # for all the 25 joints within each skeleton
    for j in range(len(body)):
        joint = body[j]

        # 1. translation to the position of robot
        trans_x = joint['x'] - cam_pos[0]
        trans_y = joint['y'] - cam_pos[1]
        trans_z = joint['z'] - cam_pos[2]

        # 2. rotation about x-axis
        rot_x = trans_x
        rot_y = sin_x * trans_z + cos_x * trans_y
        rot_z = cos_x * trans_z - sin_x * trans_y

        # 3. rotation about y-axis
        joint['x'] = cos_y * rot_x - sin_y * rot_z
        joint['y'] = rot_y
        joint['z'] = sin_y * rot_x + cos_y * rot_z


if __name__ == "__main__":
    main()
