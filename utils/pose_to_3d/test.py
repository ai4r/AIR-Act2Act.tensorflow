import cv2
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from utils.pose_to_3d.converter import convert_pose, skel_to_input
from utils.openpose.body import pose_keypoints

idx_path = './model/val_idx.npy'
data_path = './data/panoptic_dataset.pickle'


class PoseDataset(Dataset):
    def __init__(self, pickle_path):
        print("Reading data '{}'...".format(pickle_path))
        self.raw_data = []
        with open(pickle_path, 'rb') as f:
            self.raw_data = pickle.load(f)
        print('done')

        self.pairs = []

        for skel_2d, skel_3d in zip(self.raw_data['2d'], self.raw_data['3d']):
            self.pairs.append([skel_2d, skel_3d])

        self.raw_data = []  # release memory

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx_upper = [2, 0, 1, 3, 4, 5, 9, 10, 11]  # upper-body joints
        pair = self.pairs[idx]

        # [dim x joints] -> (x1,y1,x2,y2,...)
        inputs = pair[0][:, idx_upper].T.reshape(-1)  # upper-body on 2D
        outputs = pair[1][2::3, idx_upper].T.reshape(-1)  # upper-body on 3D, use only z values

        return torch.from_numpy(inputs).float(), torch.from_numpy(outputs).float()


def show_skeletons(ax_2d, ax_3d, skel_2d, z_out, z_gt=None):
    # init figures and variable
    ax_2d.clear()
    ax_3d.clear()

    edges = np.array([[0, 1], [1, 2], [1, 3], [3, 4], [4, 5], [1, 6], [6, 7], [7, 8]])

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


def dataset():
    # sample data
    val_idx = np.load(idx_path)
    val_sampler = SubsetRandomSampler(val_idx)
    pose_dataset = PoseDataset(data_path)
    val_loader = DataLoader(dataset=pose_dataset, batch_size=1, sampler=val_sampler)

    while True:
        data_iter = iter(val_loader)
        skel_2d, skel_z = next(data_iter)
        print(skel_2d)

        # inference
        z_out = convert_pose(skel_2d)

        # show
        skel_2d = skel_2d.cpu().numpy()
        skel_2d = skel_2d.reshape((2, -1), order='F')  # [(x,y) x n_joint]
        z_out = z_out.detach().cpu().numpy()
        z_out = z_out.reshape(-1)
        z_gt = skel_z.numpy().reshape(-1)
        show_skeletons(skel_2d, z_out, z_gt)


def webcam():
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

    while True:
        # get camera frame
        ret, frame = cap.read()

        # run OpenPose to get 2d pose
        key_points, output_data = pose_keypoints(frame)
        user = key_points[0]

        cv2.imshow(f'{image_width}x{image_height}', output_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if user.any():
            # convert 2d pose to 3d pose
            v_input, skel_2d = skel_to_input(user)
            z_out = convert_pose(v_input)
            z_out = z_out.detach().cpu().numpy()
            print(z_out)

            # draw 3d skeleton
            skel_2d = skel_2d.transpose()  # [(x,y) x n_joint]
            z_out = z_out.reshape(-1)

            show_skeletons(ax1, ax2, skel_2d, z_out)
            plt.show(block=False)

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()


if __name__ == "__main__":
    # dataset()
    webcam()
