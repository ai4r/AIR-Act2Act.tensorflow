import os
import math
import numpy as np
from scipy.spatial import distance

import torch
import torch.nn as nn
import torch.nn.functional as F

path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path, 'trained_net.pt')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(18, 30)
        self.fc1_bn = nn.BatchNorm1d(30)
        self.fc2 = nn.Linear(30, 20)
        self.fc2_bn = nn.BatchNorm1d(20)
        self.fc3 = nn.Linear(20, 9)

    def forward(self, x):
        x = x.view(-1, 18)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x


class OpenPoseData:
    def __init__(self):
        self.Nose = 0
        self.Neck = 1
        self.RShoulder = 2
        self.RElbow = 3
        self.RWrist = 4
        self.LShoulder = 5
        self.LElbow = 6
        self.LWrist = 7
        self.MidHip = 8

        self.upper = [self.MidHip,
                      self.Neck,
                      self.Nose,
                      self.LShoulder,
                      self.LElbow,
                      self.LWrist,
                      self.RShoulder,
                      self.RElbow,
                      self.RWrist]

        self.stand = [[+0.0, +1.7],
                      [+0.0, +0.0],
                      [+0.0, -0.5],
                      [+0.5, +0.0],
                      [+0.5, +1.0],
                      [+0.5, +2.0],
                      [-0.5, +0.0],
                      [-0.5, +1.0],
                      [-0.5, +2.0]]


# OpenPose output format
config = OpenPoseData()

# cpu or gpu?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('using device {}'.format(device))

# load net
net = Net()
net.to(device)
net.load_state_dict(torch.load(model_path))
net.train(False)


def normalize_pose(key_points):
    # remove confidence dim
    skel = key_points[:, :2]

    # resize skeleton so that the length between shoulders is 1
    rshoulder = config.RShoulder
    lshoulder = config.LShoulder
    anchor_pt = (skel[rshoulder, :] + skel[lshoulder, :]) / 2.0  # center of shoulders
    skel[1, :] = anchor_pt  # change neck point
    resize_factor = distance.euclidean(skel[rshoulder, :], skel[lshoulder, :])
    for i in range(len(skel)):
        if skel[i, :].any():
            skel[i, :] = (skel[i, :] - anchor_pt) / resize_factor

    # make shoulders align
    angle = angle_between(skel[rshoulder, :] - skel[lshoulder, :], [-1.0, 0.0])
    for i in range(len(skel)):
        x = +skel[i, 0] * math.cos(angle) + skel[i, 1] * math.sin(angle)
        y = -skel[i, 0] * math.sin(angle) + skel[i, 1] * math.cos(angle)
        skel[i, 0] = x
        skel[i, 1] = y

    return skel


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return (ang1 - ang2) % (2 * np.pi)


def pose_to_input(skel_2d):
    v_input = skel_2d.reshape(-1)
    v_input = torch.from_numpy(v_input).float()
    return v_input


def infer_z(skel_2d):
    # inference
    skel_2d = skel_2d.to(device)
    z_out = net(skel_2d)

    return z_out


def refine_pose(skel_2d):
    for idx, joint in enumerate(config.upper):
        if not skel_2d[joint].any():
            skel_2d[joint] = skel_2d[config.Neck] + config.stand[idx]
    return skel_2d
