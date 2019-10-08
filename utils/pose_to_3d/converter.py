import math
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial import distance

import torch
import torch.nn as nn
import torch.nn.functional as F

model_path = './model/trained_net.pt'


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


# cpu or gpu?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('using device {}'.format(device))

# load net
net = Net()
net.to(device)
net.load_state_dict(torch.load(model_path))
net.train(False)


def convert_pose(skel_2d):
    # inference
    skel_2d = skel_2d.to(device)
    z_out = net(skel_2d)

    return z_out


def normalize_pose(_skel):
    # remove confidence dim
    skel = _skel[:, :2]

    # resize skeleton so that the length between shoulders is 1
    anchor_pt = (skel[2, :] + skel[5, :]) / 2.0  # center of shoulders
    skel[1, :] = anchor_pt  # change neck point
    resize_factor = distance.euclidean(skel[2, :], skel[5, :])
    skel[:, :] = (skel[:, :] - anchor_pt) / resize_factor

    # make it shoulders align
    angle = angle_between(skel[2, :] - skel[5, :], [-1.0, 0.0])
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


def skel_to_input(skel_2d):
    skel_2d = normalize_pose(skel_2d)
    idx_upper = [8, 1, 0, 5, 6, 7, 2, 3, 4]
    skel_2d = skel_2d[idx_upper, :]
    v_input = skel_2d.reshape(-1)
    v_input = torch.from_numpy(v_input).float()
    return v_input, skel_2d
