# -*- coding: utf-8 -*-
"""
用于创建连接对象
"""
import json
import requests as req
from configparser import ConfigParser
from utils.loggings import log


class Connection(object):

    def __init__(self):
        self._config = ConfigParser()
        self._config_path = "config/"
        self._config.read(self._config_path + "serve.ini")
        self._object_str = "This is connection object!"

    def __str__(self):
        return self._object_str

    @property
    def config(self):
        return self._config

    @property
    def config_path(self):
        return self._config_path


class ConnectionTFServing(Connection):

    def __init__(self):
        super(ConnectionTFServing, self).__init__()
        self._ip = self._config.get("tensorflow_serving", "ip")
        self._port = self._config.get("tensorflow_serving", "port")
        self._path = self._config.get("tensorflow_serving", "path")
        self._url = "http://" + self._ip + ":" + self._port + self._path
        self._predict_result = {}
        self._object_str = "This is connection tf_serving object!"

    def __str__(self):
        return self._object_str

    def calculate_predict_result(self, x):
        """

        :param x: 待预测样本
        :return: dict 预测结果
        """
        assert isinstance(x, dict)
        self._predict_result.clear()
        try:
            r = req.post(url=self._url, data=json.dumps(x))
            log.info("[INFO] " + str(r.status_code))
            self._predict_result = r.json()
        except Exception as e:
            log.error("[ERROR] " + str(e))

    @property
    def url(self):
        return self._url

    @property
    def predict_result(self):
        return self._predict_result
