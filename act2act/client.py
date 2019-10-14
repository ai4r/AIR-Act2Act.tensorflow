import time

from utils.nao import joints_to_nao
from data.normalization import denormalize_feature
from data.constants import robot_feature_type
from act2act.test import webcam

import socket
import simplejson as json

HOST = "127.0.0.1"
CMD_PORT = 10240


def init_socket():
    # connect to NAO server
    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cmd_sock.connect((HOST, CMD_PORT))
    print("Connect to %s" % str(HOST))
    return cmd_sock


def main():
    # connect to server
    while True:
        try:
            cmd_sock = init_socket()
            break
        except socket.error:
            print("connection to server failed, retrying...")
            time.sleep(3)
    print("Server connected")

    for output in webcam():
        # send results to client
        joints = denormalize_feature(output, robot_feature_type)
        angles = joints_to_nao(joints)
        json_string = json.dumps({'target_angles': angles})
        cmd_sock.send(str(len(json_string)).ljust(16).encode('utf-8'))
        cmd_sock.sendall(json_string.encode('utf-8'))

    cmd_sock.close()


if __name__ == "__main__":
    main()
