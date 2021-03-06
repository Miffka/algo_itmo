import os.path as osp


CURRENT_PATH = osp.dirname(osp.realpath(__file__))

class SystemConfig(object):
    root_dir = osp.join(CURRENT_PATH, "..")
    data_dir = osp.realpath(osp.join(CURRENT_PATH, "..", "data"))

system_config = SystemConfig()