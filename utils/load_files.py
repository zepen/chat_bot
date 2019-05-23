# -*- coding: utf-8 -*-
"""
预加载相关文件
"""
import re
import numpy as np
from utils.loggings import log
from utils.connect import ConnectionRule

conn_rule = ConnectionRule()


class LoadFiles(object):

    def __init__(self):
        pass


class LoadCorpus(LoadFiles):

    def __init__(self):
        super(LoadCorpus, self).__init__()
        self._corpus_path = "./data/chat_corpus.txt"
        self._remove_sign = set(conn_rule.rule["remove_sign"])
        self._x_data = []
        self._load_corpus()

    def _load_corpus(self):
        with open(self._corpus_path, "r", encoding="utf-8") as f:
            pre_txt = [i.replace("\n", "") for i in f.readlines()]
        match_re = re.compile("M")
        corpus = [s for s in [
            re.sub("M ", "", i).replace("“", "").replace("”", "").replace(".", "").replace("_", "").replace("%", "").
                replace("•••••", "").replace("…", "").replace("(≧▽≦)", "").replace("⊙▽⊙", "").replace("t_t", "").
                replace("T_T", "").replace("(*¯︶¯*)", "").replace("JJ", "").replace("◑▂◐", "")
            for i in pre_txt if match_re.match(i)]]
        for n in range(int(len(corpus) / 2)):
            sentence_q = corpus[2 * n]
            sentence_a = corpus[2 * n + 1]
            rule_1 = all([len(sentence_q) >= 2, len(sentence_q) <= 30, len(sentence_a) >= 2, len(sentence_a) <= 30])
            rule_2 = all([(re.search("[a-zA-Z]",  sentence_q) is None), (re.search("[a-zA-Z]",  sentence_a) is None)])
            rule_3 = all([(re.search("[0-9]", sentence_q) is None), (re.search("[0-9]", sentence_a) is None)])
            rule_4 = len(self._remove_sign.intersection(set(list(sentence_q) + list(sentence_a)))) == 0
            if all([rule_1, rule_2, rule_3, rule_4]):
                self._x_data.append((sentence_q, sentence_a))

    @property
    def x_data(self):
        return self._x_data


class LoadDictionary(LoadFiles):
    """
    vocab_dict: 词表 (词: 序号)
    r_vocab_dict: 反词表 (序号: 词)
    """
    def __init__(self):
        super(LoadDictionary, self).__init__()
        self._vocab_dict_path = "./dictionary/vocab_dict.txt"
        self._read_vocab_dict()
        self._object_str = "This is load file object!"

    def _read_vocab_dict(self):
        with open(self._vocab_dict_path, "r", encoding="utf-8") as f:
            self._vocab_dict = f.readlines()

    def __str__(self):
        return self._object_str

    @property
    def vocab_size(self):
        return len(self._vocab_dict)


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
