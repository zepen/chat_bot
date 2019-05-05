# -*- coding: utf-8 -*-
"""
测试tf_serving是否通畅
"""
import os
from config.config import ModelConfig
from utils.load_files import LoadDictionary
from algorithm.processing import RuleCorrection
os.chdir("..")

from serve.docker_serve import predict_func, conn_tf_serving

load_dict = LoadDictionary()
m_config = ModelConfig()
rule_c = RuleCorrection()
vd = load_dict.vocab_dict
rvd = load_dict.r_vocab_dict
print("URL {}".format(conn_tf_serving.url))


def test_get_tf_serving():
    print("[Response]: {}".format(
        predict_func("你是傻逼吗？", vd, rvd, m_config, rule_c))
    )
