import glob
import os
import joblib
from sklearn.decomposition import PCA

from utils.AIR import get_upper_body_joints, read_joint, move_camera_to_front
from data.constants import src, pca_path

pca = None


def fit_pca():
    print('Fitting PCA ...')

    # Assume input data matrix X of size [N x D]
    X = list()
    files = glob.glob(src + '/*/C001*.joint')
    files.extend(glob.glob(src + '/*/C002*.joint'))
    for file in files:
        body_id = 1 if "C001" in file else 0
        body_info = read_joint(file)

        move_camera_to_front(body_info, body_id=body_id)

        for f in range(len(body_info)):
            body = body_info[f][body_id]["joints"]
            features = get_features(body)
            X.append(features)

    pca = PCA(n_components=0.999)
    pca.fit(X)
    return pca


def get_features(body):
    joints = get_upper_body_joints(body)
    features = [v for joint in joints for v in joint]
    return features


def get_pca():
    if not os.path.exists(pca_path):
        pca = fit_pca()
        joblib.dump(pca, pca_path)
        print('Save PCA to ', pca_path)
    else:
        pca = joblib.load(pca_path)
        print('Load PCA from ', pca_path)

    print('# of components of pca: ', pca.n_components_)
    # print('explained variance ratio: ', pca.explained_variance_ratio_)
    return pca


def transform(features):
    global pca
    if pca is None:
        pca = get_pca()
    return pca.transform(features)


def inverse_transform(reduced):
    global pca
    if pca is None:
        pca = get_pca()
    return pca.inverse_transform(reduced)


def n_components():
    global pca
    if pca is None:
        pca = get_pca()
    return pca.n_components_


def test():
    file = './joint files/A005/C001P001A005S001.joint'
    body_info = read_joint(file)
    body = body_info[0][1]["joints"]
    features = [get_features(body)]
    print('original: ', features)

    pca = get_pca()
    reduced = pca.transform(features)
    print('transformed: ', reduced)

    restored = pca.inverse_transform(reduced)
    print('restored: ', restored)


if __name__ == '__main__':
    test()
