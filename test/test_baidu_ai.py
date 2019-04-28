# -*- coding: utf-8 -*-
"""
测试baidu_ai_api
"""
import os
from algorithm.language_model import BaiduDnnLM

os.chdir("..")


def test_baidu_dnn_lm():
    text = "我从山中来，带着兰花草！"
    baidu_dnn_lm = BaiduDnnLM()
    print(baidu_dnn_lm(text=text))
