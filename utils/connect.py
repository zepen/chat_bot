# -*- coding: utf-8 -*-
"""
读取连接信息
"""
import json
from configparser import ConfigParser


class Connection(object):

    def __init__(self):
        self._config = ConfigParser()
        self._config_path = "./config/"
        self._object_str = "[INFO] This is connection object!"

    def __str__(self):
        return self._object_str

    @property
    def config(self):
        return self._config

    @property
    def config_path(self):
        return self._config_path


class ConnectionRule(Connection):

    def __init__(self):
        super(ConnectionRule, self).__init__()
        with open(self._config_path + "rule.json", "r", encoding="utf-8") as f:
            self._rule = json.load(f)
        self._object_str = "[INFO] This is connection rule object!"

    def __str__(self):
        return self._object_str

    @property
    def rule(self):
        return self._rule


class ConnectionTFServing(Connection):

    def __init__(self):
        super(ConnectionTFServing, self).__init__()
        self._config.read(self._config_path + "tf_serving.ini")
        self._ip = self._config.get("tensorflow_serving", "ip")
        self._port = self._config.get("tensorflow_serving", "port")
        self._path = self._config.get("tensorflow_serving", "path")
        self._name = self._config.get("tensorflow_serving", "name")
        self._url = "http://" + self._ip + ":" + self._port + self._path
        self._object_str = "[INFO] This is connection tf_serving object!"

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


class ConnectionNeo4j(Connection):

    def __init__(self):
        super(ConnectionNeo4j, self).__init__()
        self._config.read(self._config_path + "neo4j.ini")
        self._ip = self._config.get("neo4j_serving", "ip")
        self._port_1 = self._config.get("neo4j_serving", "port_1")
        self._port_2 = self._config.get("neo4j_serving", "port_2")
        self._port_3 = self._config.get("neo4j_serving", "port_3")
        self._username = self._config.get("neo4j_serving", "user")
        self._password = self._config.get("neo4j_serving", "passwd")
        self._name = self._config.get("neo4j_serving", "name")
        self._object_str = "[INFO] This is connection neo4j object!"

    def __str__(self):
        return self._object_str

    @property
    def ip(self):
        return self._ip

    @property
    def port_1(self):
        return self._port_1

    @property
    def port_2(self):
        return self._port_2

    @property
    def port_3(self):
        return self._port_3

    @property
    def username(self):
        return self._username

    @property
    def password(self):
        return self._password

    @property
    def name(self):
        return self._name


class ConnectBaiduAI(Connection):

    def __init__(self):
        super(ConnectBaiduAI, self).__init__()
        self._config.read(self._config_path + "baidu_ai.ini")
        self._app_id = self._config.get("baidu_ai", "app_id")
        self._api_key = self._config.get("baidu_ai", "api_key")
        self._secret_key = self._config.get("baidu_ai", "secret_key")
        self._object_str = "[INFO] This is connection baidu_ai object!"

    def __str__(self):
        return self._object_str

    @property
    def app_id(self):
        return self._app_id

    @property
    def api_key(self):
        return self._api_key

    @property
    def secret_key(self):
        return self._secret_key
