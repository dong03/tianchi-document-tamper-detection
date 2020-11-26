# -*- coding: utf-8 -*-

def _init():  # 初始化
    global _global_dict
    _global_dict = {}


def set_value(key, value):
    """ 定义一个全局变量 """
    _global_dict[key] = value

def update_global(config):
    """ 定义一个全局变量 """
    _global_dict['max_anchors_size'] = int(config[type]['imageSize'])
    _global_dict['min_anchors_size'] = int(config[type]['imageSize'])
    _global_dict['stride'] = int(config[type]['stride'])
    _global_dict['anchors'] = [(_global_dict['max_anchors_size'], _global_dict['max_anchors_size']),
                               (_global_dict['min_anchors_size'], _global_dict['min_anchors_size'])]


def get_value(key, defValue=None):
    return _global_dict[key]
