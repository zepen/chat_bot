# -*- coding: utf-8 -*-
"""
预加载相关文件
"""
from pickle import load
from utils.loggings import log


class LoadFiles(object):
    """
    vocab_dict: 词表 (词: 序号)
    r_vocab_dict: 反词表 (序号: 词)
    """
    def __init__(self):
        self._vocab_dict = {}
        self._r_vocab_dict = {}
        self._dict_path = "./dictionary/"
        self._vocab_dict_path = self._dict_path + "vocab_dict.pkl"
        self._r_vocab_dict_path = self._dict_path + "r_vocab_dict.pkl"
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
        with open(self._r_vocab_dict_path, "rb") as f:
            self._r_vocab_dict = load(f)
        log.info("[INFO] The r_vocab_dict is load!")

    @property
    def vocab_dict(self):
        return self._vocab_dict

    @property
    def r_vocab_dict(self):
        return self._r_vocab_dict
