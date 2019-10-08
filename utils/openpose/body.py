# From Python
# It requires OpenCV installed for Python
import sys
import os
from sys import platform

openpose_path = "C:/Users/wrko/Desktop/openpose"
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(openpose_path + '/build/python/openpose/Release')
        os.environ['PATH'] += ';' + openpose_path + '/build/x64/Release;'
        os.environ['PATH'] += ';' + openpose_path + '/build/bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(openpose_path + '/python')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = openpose_path + "/models/"

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


def pose_keypoints(image):
    # Process Image
    datum = op.Datum()
    datum.cvInputData = image
    opWrapper.emplaceAndPop([datum])

    # Return Results
    return datum.poseKeypoints, datum.cvOutputData
