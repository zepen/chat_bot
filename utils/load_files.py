# -*- coding: utf-8 -*-
"""
预加载相关文件
"""
import re
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
        corpus = [s for s in [re.sub("M ", "", i) for i in pre_txt if match_re.match(i)]]
        for n in range(int(len(corpus) / 2)):
            sentence_q = corpus[2 * n]
            sentence_a = corpus[2 * n + 1]
            rule_1 = ((len(sentence_q) >= 2) and (len(sentence_q) <= 30)) and \
                     ((len(sentence_a) >= 2) and (len(sentence_a) <= 30))
            rule_2 = (re.search("[a-zA-Z]",  sentence_q) is None) and (re.search("[a-zA-Z]",  sentence_a) is None)
            rule_3 = (re.search("[0-9]", sentence_q)is None) and (re.search("[0-9]", sentence_a) is None)
            rule_4 = len(self._remove_sign.intersection(set(list(sentence_q) + list(sentence_q)))) == 0
            if rule_1 and rule_2 and rule_3 and rule_4:
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
        self._rule_dict = conn_rule.rule["replace_text"]
        log.info("[INFO] The rule file is load!")

    def __call__(self, output_text):
        return self.rule_correction_fun(output_text)

    def rule_correction_fun(self, output_text: str) -> str:
        """  对模型预测文本结果进行校正
        :param output_text: 预测文本
        :return: str
        """
        return self._rule_dict[output_text] \
            if self._rule_dict.get(output_text) else output_text
