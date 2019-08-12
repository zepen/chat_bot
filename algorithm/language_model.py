# -*- coding:utf-8 -*-
"""
用语言模型来评估回复结果
"""
import kenlm
from aip import AipNlp
from utils.connect import ConnectBaiduAI


class LanguageModel(object):

    def __init__(self):
        self._object_str = "[INFO] This is lm object!"

    def __str__(self):
        return self._object_str


class BaiduDnnLM(LanguageModel):

    def __init__(self):
        super(BaiduDnnLM, self).__init__()
        conn_baidu_ai = ConnectBaiduAI()
        self._client = AipNlp(conn_baidu_ai.app_id, conn_baidu_ai.api_key, conn_baidu_ai.secret_key)

    def __call__(self, *args, **kwargs):
        return self._client.dnnlm(kwargs["text"])
