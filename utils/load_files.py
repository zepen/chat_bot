# -*- coding: utf-8 -*-
"""
预加载相关文件
"""
import re
from pickle import load
from utils.loggings import log


class LoadFiles(object):

    def __init__(self):
        pass


class LoadCorpus(LoadFiles):

    def __init__(self):
        super(LoadCorpus, self).__init__()
        self._corpus_path = "./corpus/chat_corpus.txt"
        self._load_corpus()

    def _load_corpus(self):
        with open(self._corpus_path, "r", encoding="utf-8") as f:
            pre_txt = [i.replace("\n", "") for i in f.readlines()]
        match_re = re.compile("M")
        corpus = [s.split("/") for s in [re.sub("M ", "", i) for i in pre_txt if match_re.match(i)]]
        self._x_data = [(corpus[2 * n], corpus[2 * n + 1]) for n in range(int(len(corpus) / 2))]

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
        self._vocab_dict = {}
        self._r_vocab_dict = {}
        self._vocab_dict_path = "./dictionary/vocab_dict.pkl"
        self._object_str = "This is load file object!"
        self._load_vocab_dict()
        self._r_load_vocab_dict()

    def __str__(self):
        return self._object_str

    def _load_vocab_dict(self):
        with open(self._vocab_dict_path, "rb") as f:
            self._vocab_dict = load(f)
        log.info("[INFO] The vocab_dict is load!")

    def _r_load_vocab_dict(self):
        self._r_vocab_dict = {v: k for k, v in self.vocab_dict.items()}
        log.info("[INFO] The r_vocab_dict is load!")

    @property
    def vocab_dict(self):
        return self._vocab_dict

    @property
    def r_vocab_dict(self):
        return self._r_vocab_dict
