import numpy as np
from math import cos, sin, atan2, fabs, copysign
from utils.AIR import get_upper_body_joints


def convert_to_nao(body):
    return joints_to_nao(get_upper_body_joints(body))


def joints_to_nao(joints):
    # joint information
    spineBase, spineShoulder, head, shoulderLeft, elbowLeft, wristLeft, shoulderRight, elbowRight, wristRight = joints

    ####### RIGHT ARM #######
    r_8_9_human = elbowRight - shoulderRight
    r_9_10_human = wristRight - elbowRight
    RShoulderPitch = atan2(-r_8_9_human[1], -r_8_9_human[2])

    r_8_9_rsp = rotate_x(r_8_9_human, -RShoulderPitch)
    r_9_10_rsp = rotate_x(r_9_10_human, -RShoulderPitch)
    RShoulderRoll = atan2(-r_8_9_rsp[0], -r_8_9_rsp[2])

    r_9_10_rsr = rotate_y(r_9_10_rsp, RShoulderRoll)
    new_y = np.cross(r_9_10_rsr, [0, 0, 1])
    RElbowYaw = atan2(new_y[0], new_y[1])

    r_9_10_rey = rotate_z(r_9_10_rsr, -RElbowYaw)
    RElbowRoll = atan2(-r_9_10_rey[0], -r_9_10_rey[2])

    ####### LEFT ARM #######
    r_4_5_human = elbowLeft - shoulderLeft
    r_5_6_human = wristLeft - elbowLeft
    LShoulderPitch = atan2(-r_4_5_human[1], -r_4_5_human[2])

    r_4_5_rsp = rotate_x(r_4_5_human, -LShoulderPitch)
    r_5_6_rsp = rotate_x(r_5_6_human, -LShoulderPitch)
    LShoulderRoll = atan2(-r_4_5_rsp[0], -r_4_5_rsp[2])

    r_5_6_rsr = rotate_y(r_5_6_rsp, LShoulderRoll)
    new_y = np.cross([0, 0, 1], r_5_6_rsr)
    LElbowYaw = atan2(new_y[0], new_y[1])

    r_5_6_rey = rotate_z(r_5_6_rsr, -LElbowYaw)
    LElbowRoll = atan2(-r_5_6_rey[0], -r_5_6_rey[2])

    ####### BODY #######
    r_0_20_human = spineShoulder - spineBase
    r_g_human = [0, 1, 0]
    LHipYawPitch = rotation_angle([r_g_human[2], r_g_human[1]], [r_0_20_human[2], r_0_20_human[1]])
    # LHipYawPitch += radians(-14.5)

    ####### HEAD #######
    r_20_3_human = head - spineShoulder
    r_0_20_human = spineShoulder - spineBase
    HeadPitch = rotation_angle([r_0_20_human[2], r_0_20_human[1]], [r_20_3_human[2], r_20_3_human[1]])
    # HeadPitch += radians(6.5)

    return [LHipYawPitch, HeadPitch,
            LShoulderPitch, LShoulderRoll, LElbowYaw, LElbowRoll,
            RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll]


# (mean, min, max) values of joints in radians
def configuration():
    return [(0.0, -1.1453, 0.7408), (0.0, -0.6720, 0.5149),
            (1.5908, -2.0857, 2.0857), (0.0, -0.3142, 1.3265), (0.0, -2.0857, 2.0857), (0.0, -1.5446, -0.0349),
            (1.5908, -2.0857, 2.0857), (0.0, -1.3265, 0.3142), (0.0, -2.0857, 2.0857), (0.0, 0.0349, 1.5446)]


def rotate_x(vector, angle):
    A = np.array([[1, 0, 0],
                  [0, cos(angle), sin(angle)],
                  [0, -sin(angle), cos(angle)]])
    return np.dot(A, vector)


def rotate_y(vector, angle):
    A = np.array([[cos(angle), 0, -sin(angle)],
                  [0, 1, 0],
                  [sin(angle), 0, cos(angle)]])
    return np.dot(A, vector)


def rotate_z(vector, angle):
    A = np.array([[cos(angle), sin(angle), 0],
                  [-sin(angle), cos(angle), 0],
                  [0, 0, 1]])
    return np.dot(A, vector)


# rotation_angle from v1 to v2 (counterclockwise) in (-pi, pi]
def rotation_angle(v1, v2):
    angle_v1 = atan2(v1[1], v1[0])
    angle_v2 = atan2(v2[1], v2[0])
    angle_between = angle_v2 - angle_v1
    if fabs(angle_between) >= np.pi:
        angle_between -= copysign(2 * np.pi, angle_between)
    return angle_between


# solve kinematics of humanoid robot using DH parameters (only upper body)
# x-axis: up, y-axis: front, z-axis: right
def solve_kinematics(LHipYawPitch, HeadPitch, LShoulderPitch, LShoulderRoll, LElbowYaw, LElbowRoll,
                     RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll):
    # pelvis (origin)
    pelvis = np.array([0, 0, 0])

    # head
    T_neck_pelvis = transform(3, 0, 0, LHipYawPitch)
    neck = T_neck_pelvis[:3, -1]

    T_head_neck = transform(1, 0, 0, HeadPitch)
    T_head_pelvis = np.matmul(T_neck_pelvis, T_head_neck)
    head = T_head_pelvis[:3, -1]

    # left arm
    T_lshoulder_neck = transform(0, np.pi / 2, 0.7, LShoulderPitch + np.pi / 2)
    T_lshoulder_pelvis = np.matmul(T_neck_pelvis, T_lshoulder_neck)
    lshoulder = T_lshoulder_pelvis[:3, -1]

    T_lelbow_lshoulder = np.dot(transform(0, np.pi / 2, 0, LShoulderRoll + np.pi / 2),
                                transform(0, -np.pi / 2, 1, LElbowYaw))
    T_lelbow_pelvis = np.matmul(T_lshoulder_pelvis, T_lelbow_lshoulder)
    lelbow = T_lelbow_pelvis[:3, -1]

    T_lwrist_lelbow = transform(1, 0, 0, LElbowRoll - np.pi / 2)
    T_lwrist_pelvis = np.matmul(T_lelbow_pelvis, T_lwrist_lelbow)
    lwrist = T_lwrist_pelvis[:3, -1]

    # right arm
    T_rshoulder_neck = transform(0, np.pi / 2, -0.7, RShoulderPitch + np.pi / 2)
    T_rshoulder_pelvis = np.matmul(T_neck_pelvis, T_rshoulder_neck)
    rshoulder = T_rshoulder_pelvis[:3, -1]

    T_relbow_rshoulder = np.dot(transform(0, np.pi / 2, 0, RShoulderRoll + np.pi / 2),
                                transform(0, -np.pi / 2, 1, RElbowYaw))
    T_relbow_pelvis = np.matmul(T_rshoulder_pelvis, T_relbow_rshoulder)
    relbow = T_relbow_pelvis[:3, -1]

    T_rwrist_relbow = transform(1, 0, 0, RElbowRoll - np.pi / 2)
    T_rwrist_pelvis = np.matmul(T_relbow_pelvis, T_rwrist_relbow)
    rwrist = T_rwrist_pelvis[:3, -1]

    def change_coord(vector):
        return np.array([-vector[2], vector[0], -vector[1]])
    return tuple([change_coord(joint) for joint in (pelvis, neck, head, lshoulder, lelbow, lwrist, rshoulder, relbow, rwrist)])


def transform(a, alpha, d, theta):
    return np.array([[cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
                      [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                      [0, sin(alpha), cos(alpha), d],
                      [0, 0, 0, 1]])
