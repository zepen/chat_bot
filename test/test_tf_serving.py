# -*- coding: utf-8 -*-
"""
测试tf_serving是否通畅
"""
import os
from config.config import ModelConfig
os.chdir("..")

from utils.load_files import LoadDictionary, RuleCorrection
from serve.docker_serve import predict_func, conn_tf_serving

load_dict = LoadDictionary()
m_config = ModelConfig()
rule_c = RuleCorrection()
print("URL {}".format(conn_tf_serving.url))


def test_get_tf_serving():
    # 请求1000次，查看返回结果
    for _ in range(1000):
        print("[Response]: {}".format(
            predict_func("今天天气很好!"))
        )
