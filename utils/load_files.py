# -*- coding: utf-8 -*-
"""
预加载相关文件
"""
import re


class LoadFiles(object):

    def __init__(self):
        pass


class LoadCorpus(LoadFiles):

    def __init__(self):
        super(LoadCorpus, self).__init__()
        self._corpus_path = "./data/chat_corpus.txt"
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
