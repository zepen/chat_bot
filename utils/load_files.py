# -*- coding: utf-8 -*-
"""
预加载相关文件
"""
import re
import numpy as np
from utils.loggings import log
from utils.connect import ConnectionRule

conn_rule = ConnectionRule()


class RuleCorrection(object):

    def __init__(self):
        self._load_rule()

    def _load_rule(self):
        self._replace_name = conn_rule.rule["replace_name"]
        self._replace_content = conn_rule.rule["replace_content"]
        log.info("[INFO] The rule file is load!")

    def replace_name(self, response: str) -> str:
        """ 替换机器人名称

        :param response: 模型生成文本
        :return:
        """
        for k in self._replace_name.keys():
            if re.search(k, response):
                return re.sub(k, self._replace_name[k], response)
        return response

    def replace_content(self) -> str:
        """  替换困惑度过高的语句

        :return: str
        """
        replace_content_len = len(self._replace_content)
        random_index = np.random.randint(0, replace_content_len)
        return self._replace_content[random_index]
