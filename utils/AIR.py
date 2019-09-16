import os
import numpy as np
import simplejson as json


def read_joint(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fp:
            data = fp.read()
            body_info = json.loads(data)
    except:
        print('Error occurred: ' + path)
    return body_info


def write_joint(path, body_info):
    with open(path, 'w') as fp:
        json.dump(body_info, fp, indent=2)


def vectorize(joint):
    return np.array([joint['x'], joint['y'], joint['z']])


def get_upper_body_joints(body):
    shoulderRight = vectorize(body[8])
    shoulderLeft = vectorize(body[4])
    elbowRight = vectorize(body[9])
    elbowLeft = vectorize(body[5])
    wristRight = vectorize(body[10])
    wristLeft = vectorize(body[6])

    torso = vectorize(body[0])
    spineShoulder = vectorize(body[20])
    head = vectorize(body[3])

    return torso, spineShoulder, head, shoulderLeft, elbowLeft, wristLeft, shoulderRight, elbowRight, wristRight


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
