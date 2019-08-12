# -*- coding:utf-8 -*-
"""
定义实体抽取模型
"""
from pyltp import Segmentor, Postagger, NamedEntityRecognizer


class NerModel(object):

    def __init__(self):
        self._object_str = "[INFO] This is ner object!"

    def __str__(self):
        return self._object_str


class BirLstmCrf(NerModel):

    def __init__(self):
        super().__init__()
        self._object_str = "[INFO] This is bir_lstm_crf object!"

    def __str__(self):
        return self._object_str


class Ltp(NerModel):

    def __init__(self):
        super(Ltp, self).__init__()
        self._model_path = "./model/ltp/"
        self._seg = Segmentor()
        self._pos = Postagger()
        self._recognizer = NamedEntityRecognizer()
        self._load_model()
        self._object_str = "[INFO] This is ltp object!"
        print("[INFO] All model is load!")

    def __str__(self):
        return self._object_str

    def _load_model(self):
        self._seg.load(self._model_path + "cws.model")
        self._pos.load(self._model_path + "pos.model")
        self._recognizer.load(self._model_path + "ner.model")

    def get_entity(self, sentence):
        words = self._seg.segment(sentence)
        pos = self._pos.postag(words)
        ner = self._recognizer.recognize(words, pos)
        entity = [w for w, s in zip(words, ner) if s != 'O']
        if entity:
            return "".join(entity) if len(entity) > 1 else entity[0]
