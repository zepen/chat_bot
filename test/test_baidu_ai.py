# -*- coding: utf-8 -*-
"""
测试baidu_ai_api
"""
from algorithm.language_model import BaiduDnnLM


def test_baidu_dnn_lm():
    text = "我从山中来，带着兰花草！"
    baidu_dnn_lm = BaiduDnnLM()
    print("ppl: {}".format(baidu_dnn_lm.get_ppl(text=text)))
