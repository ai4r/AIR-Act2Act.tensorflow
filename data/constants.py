import os

actions = ["A001",  # bow
           "A002",  # look
           "A005",  # handshake
           "A006"]  # front both arms

source_len = 30
target_len = 20

context_len = 10
dist_len = 1

human_feature_type = 'torso'  # {'torso', 'nao_angles'}
robot_feature_type = 'nao_angles'  # {'nao_angles'}

src = "./joint files"
dst_train = os.path.normpath(os.path.join('./extracted files',
    'human_{0}'.format(human_feature_type), 'robot_{0}'.format(robot_feature_type),
    'in_{0}'.format(source_len), 'out_{0}'.format(target_len), 'train'))
dst_test = os.path.normpath(os.path.join('./extracted files',
    'human_{0}'.format(human_feature_type), 'robot_{0}'.format(robot_feature_type),
    'in_{0}'.format(source_len), 'out_{0}'.format(target_len), 'test'))
