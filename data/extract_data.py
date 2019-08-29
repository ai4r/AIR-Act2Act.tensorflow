import glob
import random
import numpy as np
from tqdm import tqdm

from data.constants import *
from data.normalization import normalize_body_data
from utils.AIR import vectorize, read_joint

kinect_frame_rate = 30  # frame rate of kinect camera
target_frame_rate = 10  # frame rate of extracted data

b_iter = False  # whether to iterate the first frame
max_distance = 5.  # maximum distance between camera and human
p_test = 0.1  # probability to be used as test data
p_data = 1.0  # probability to be selected as data from AIR-Act2Act DB


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
            pred_len = 1 if b_test else target_len

            seq_generator = generate_seq(human_info, robot_info, third_info, pred_len=pred_len)

            index = 0
            for human_seq, robot_seq in seq_generator:
                data_name = human_file.replace("\\", "/").split("/")[-1].split('.')[0]
                data_file = dst + "/{}_{:03d}.npz".format(data_name, index)
                np.savez(data_file, human_seq=human_seq, robot_seq=robot_seq)
                index += 1

            pbar.update(1)

        pbar.close()


# generate the sequence of features to be used as training & test data.
def generate_seq(human, robot, third, pred_len=1):
    # extract distance features first
    n_seq = min(len(human), len(robot), len(third))
    step = round(kinect_frame_rate / target_frame_rate)
    third_features = []
    for f in range(0, n_seq, step):
        itr = source_len - 1 if f == 0 and b_iter else 1  # iterate the first frame
        for _ in range(itr):
            # distance features
            n_body = sum(1 for b in third[f] if b is not None)
            if n_body != 2:
                print('third camera information is wrong.')
                continue
            robot_pos1 = vectorize(third[f][0]["joints"][0])
            human_pos1 = vectorize(third[f][1]["joints"][0])
            dist_third = max_distance if all(v == 0 for v in human_pos1) else np.linalg.norm(human_pos1 - robot_pos1)

            human_pos2 = vectorize(human[f][1]["joints"][0])
            robot_pos2 = np.array([0., 0., 0.])
            dist_human = max_distance if all(v == 0 for v in human_pos2) else np.linalg.norm(human_pos2 - robot_pos2)

            dist = min(dist_third, dist_human)
            third_features.append([dist / max_distance])

    # move camera position in front of person
    move_camera_to_front(human, body_id=1)
    move_camera_to_front(robot, body_id=0)

    # extract features of the human & robot body
    human_features = []
    robot_features = []
    for f in range(0, n_seq, step):
        itr = source_len - 1 if f == 0 and b_iter else 1  # iterate the first frame
        for _ in range(itr):
            human_features.append(normalize_body_data(human[f][1]["joints"], human_feature_type))
            robot_features.append(normalize_body_data(robot[f][0]["joints"], robot_feature_type))

    # extract sequences
    for start in range(len(human_features) - source_len - (pred_len - 1)):
        if all(v == [1.0] for v in third_features[start:start + source_len]):
            continue
        human_seq = [third_features[i] + human_features[i] for i in range(start, start + source_len           )]
        robot_seq = [                    robot_features[i] for i in range(start, start + source_len + pred_len)]
        yield human_seq, robot_seq


def move_camera_to_front(body_info, body_id):
    for f in range(len(body_info)):
        # joints of the trunk
        reference_body = body_info[f][body_id]["joints"]
        r_16_kinect = vectorize(reference_body[16])
        r_12_kinect = vectorize(reference_body[12])
        r_20_kinect = vectorize(reference_body[20])
        r_0_kinect = vectorize(reference_body[0])
        dist_to_camera = np.linalg.norm(r_0_kinect)

        # find the front direction vector
        front_vector = np.cross(r_16_kinect - r_12_kinect, r_16_kinect - r_20_kinect)
        norm = np.linalg.norm(front_vector, axis=0, ord=2)
        norm = np.finfo(front_vector.dtype).eps if norm == 0 else norm
        normalized_front_vector = front_vector / norm * dist_to_camera
        cam_pos = r_0_kinect + normalized_front_vector
        cam_dir = -normalized_front_vector
        if all(x != 0 for x in cam_dir):
            start_frame = f
            break

    for f in range(start_frame, len(body_info)):
        body = body_info[f][body_id]["joints"]

        # rotation factors
        dist = np.linalg.norm(cam_dir)
        dist_y = np.linalg.norm(np.array([cam_dir[0], cam_dir[2]]))
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

    return body_info


if __name__ == "__main__":
    main()
