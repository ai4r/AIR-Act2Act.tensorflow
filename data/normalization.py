import numpy as np
import math

from utils.nao import convert_to_nao, solve_kinematics
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
    return ((nao_angles + math.pi) / (math.pi * 2)).tolist()


# convert nao angles to body 3d positions
def denorm_from_nao_angles(features):
    angles = features * (math.pi * 2) - math.pi
    return solve_kinematics(*angles)
