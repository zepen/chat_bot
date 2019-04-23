# -*- coding: utf-8 -*-
"""
读取连接信息
"""
from configparser import ConfigParser


class Connection(object):

    def __init__(self):
        self._config = ConfigParser()
        self._config_path = "./config/"
        self._config.read(self._config_path + "tf_serving.ini")
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
        self._name = self._config.get("tensorflow_serving", "name")
        self._url = "http://" + self._ip + ":" + self._port + self._path
        self._object_str = "This is connection tf_serving object!"

    def __str__(self):
        return self._object_str

    @property
    def url(self):
        return self._url

    @property
    def port(self):
        return self._port

    @property
    def name(self):
        return self._name
