import numpy as np
import math

from utils.nao import convert_to_nao, solve_kinematics, configuration
from utils.AIR import get_upper_body_joints
from data.pca import pca


def normalize_body_data(body, feature_type):
    if feature_type == 'nao_angles':
        return norm_to_nao_angles(body)
    elif feature_type == 'torso':
        return norm_to_torso(body)
    elif feature_type == 'pca':
        return norm_by_pca(body)
    else:
        raise (ValueError, "unknown type to normalize: %s" % feature_type)


def denormalize_feature(features, feature_type):
    if feature_type == 'nao_angles':
        return denorm_from_nao_angles(features)
    elif feature_type == 'torso':
        return denorm_from_torso(features)
    elif feature_type == 'pca':
        return denorm_from_pca(features)
    else:
        raise (ValueError, "unknown type to normalize: %s" % feature_type)


def count_feature(feature_type):
    if feature_type == 'nao_angles':
        return 10
    elif feature_type == 'torso':
        return 24
    elif feature_type == 'pca':
        return pca.n_components()
    else:
        raise (ValueError, "unknown type to normalize: %s" % feature_type)


# move origin to torso and
# normalize to the distance between torso and spineShoulder
def norm_to_torso(body):
    torso, spineShoulder, head, shoulderLeft, elbowLeft, wristLeft, shoulderRight, elbowRight, wristRight = \
        get_upper_body_joints(body)

    features = list()
    features.extend(norm_to_distance(torso, spineShoulder, spineShoulder))
    features.extend(norm_to_distance(torso, spineShoulder, head))
    features.extend(norm_to_distance(torso, spineShoulder, shoulderLeft))
    features.extend(norm_to_distance(torso, spineShoulder, elbowLeft))
    features.extend(norm_to_distance(torso, spineShoulder, wristLeft))
    features.extend(norm_to_distance(torso, spineShoulder, shoulderRight))
    features.extend(norm_to_distance(torso, spineShoulder, elbowRight))
    features.extend(norm_to_distance(torso, spineShoulder, wristRight))
    return features


def norm_to_distance(origin, basis, joint):
    return (joint - origin) / np.linalg.norm(basis - origin) if any(joint) else joint


def denorm_from_torso(features):
    # pelvis
    pelvis = np.array([0, 0, 0])

    # other joints (spineShoulder, head, shoulderLeft, elbowLeft, wristLeft, shoulderRight, elbowRight, wristRight)
    spine_len = 3.
    features = features * spine_len

    return np.vstack((pelvis, np.split(features, 8)))


def norm_by_pca(body):
    features = pca.get_features(body)
    return pca.transform([features])[0]


def denorm_from_pca(features):
    restored = pca.inverse_transform([features])[0]
    joints = np.split(restored, 9)

    pelvis = np.array([0, 0, 0])
    features = list()
    for idx in range(1, len(joints)):
        features.append(norm_to_distance(joints[0], joints[1], joints[idx]))
    spine_len = 3.
    features = np.array(features) * spine_len

    return np.vstack((pelvis, features))


# convert body 3d positions to nao angles
def norm_to_nao_angles(body):
    nao_angles = np.array(convert_to_nao(body))
    config = configuration()

    normalized = []
    for idx in range(len(nao_angles)):
        nao_angle = nao_angles[idx]
        # normalize to [0, 1]
        min = config[idx][0] - np.pi
        max = config[idx][0] + np.pi
        if nao_angle < min:
            nao_angle += np.pi * 2
        if nao_angle > max:
            nao_angle -= np.pi * 2
        value = (nao_angle - min) / (max - min)
        # scale to [-1, 1]
        value = value * (1 - (-1)) + (-1)
        normalized.append(value)

    return normalized


# convert nao angles to body 3d positions
def denorm_from_nao_angles(features):
    config = configuration()

    angles = []
    for idx in range(len(features)):
        # scale to [0, 1]
        value = (features[idx] - (-1)) / (1 - (-1))
        # de-normalize to original scale
        min = config[idx][0] - np.pi
        max = config[idx][0] + np.pi
        value = value * (max - min) + min
        if value < min:
            value += np.pi * 2
        if value > max:
            value -= np.pi * 2
        angles.append(value)

    return solve_kinematics(*angles)
