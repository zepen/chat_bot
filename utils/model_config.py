"""
定义模型配置
"""
from algorithm.processing import RuleCorrection


class ModelConfig(object):
    max_decode_len = 30
    replace_sentence = "I am NLAI"
    rule_correction = RuleCorrection()
