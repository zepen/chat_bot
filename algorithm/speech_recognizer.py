# -*- coding:utf-8 -*-
"""
调用百度API获取语音识别
"""
from aip import AipSpeech
from utils.connect import ConnectionRule, ConnectBaiduAI


class SpeechRecognizer(object):

    def __init__(self):
        conn_rule = ConnectionRule()
        self.voice_reply = conn_rule.rule["voice_reply"]
        self._object_str = "[INFO] This is sr object!"

    def __repr__(self):
        return self._object_str


class BaiduSpeechRecognizer(SpeechRecognizer):

    def __init__(self):
        super(BaiduSpeechRecognizer, self).__init__()
        conn_baidu_ai = ConnectBaiduAI()
        self._client = AipSpeech(conn_baidu_ai.app_id, conn_baidu_ai.api_key, conn_baidu_ai.secret_key)

    def __call__(self, *args, **kwargs):
        return self._client.asr(kwargs["voice"], 'wav', 16000, {'dev_pid': 1536})
