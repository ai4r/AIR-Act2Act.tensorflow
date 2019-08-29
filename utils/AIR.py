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
