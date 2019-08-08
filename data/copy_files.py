### copy joint files from  ###

import glob
import shutil
import os
from constants import actions

src = r"D:\HRI DB\AIR-Act2Act-1"
dst = r".\joint files"


def main():
    for action in actions:
        joint_files = glob.glob(src + '/*/*' + action + '*.joint')
        for joint_file in joint_files:
            name = joint_file.split('\\')[-1]
            new_folder = dst + '/' + action
            if not os.path.exists(new_folder):
                os.mkdir(new_folder)
            new_file = new_folder + '/' + name
            shutil.copy(joint_file, new_file)
        print(action + ": complete")


if __name__ == "__main__":
    main()
