# -*- coding:utf-8 -*-
"""
用语言模型来评估回复结果
"""
# import kenlm
from aip import AipNlp
from utils.connect import ConnectBaiduAI


class LanguageModel(object):

    def __init__(self):
        self._object_str = "[INFO] This is lm object!"

    def __repr__(self):
        return self._object_str


class BaiduDnnLM(LanguageModel):

    def __init__(self):
        super(BaiduDnnLM, self).__init__()
        conn_baidu_ai = ConnectBaiduAI()
        self._client = AipNlp(conn_baidu_ai.app_id, conn_baidu_ai.api_key, conn_baidu_ai.secret_key)

    def __repr__(self):
        return "<This is Baidu NLP API>"

    def get_reply(self, text):
        """  获取接口回复
        :param text: 输入文本内容
        :return: 百度API返回结果，dict
        """
        return self._client.dnnlm(text=text)

    def get_ppl(self, text):
        """  获取句子混淆度
        :param text: 输入文本内容
        :return: 句子混淆度，混淆度越小，句子越符合语言逻辑
        """
        return self._client.dnnlm(text=text)["ppl"]
