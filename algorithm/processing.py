# -*- coding:utf-8 -*-
"""
Processing Data
"""
import re
import json
import numpy as np
from utils.loggings import log
from utils.load_files import LoadDictionary, LoadCorpus
from utils.connect import ConnectionRule


class ProcessingCorps(object):

    def __init__(self):
        load_dictionary = LoadDictionary()
        self._vocab_dict = load_dictionary.vocab_dict
        self._r_vocab_dict = load_dictionary.r_vocab_dict
        load_corpus = LoadCorpus()
        self._x_data = load_corpus.x_data
        print("[INFO] Data size is {}.".format(len(self._x_data)))
        self._clear_data()
        print("[INFO] Data size is {} after clear.".format(len(self._x_data)))
        conn_rule = ConnectionRule()
        self._remove_sign = set(conn_rule.rule["remove_sign"])

    def _clear_data(self):
        for sen_pair in self._x_data:
            if len(sen_pair[0]) < 2 or len(sen_pair[0]) > 30 or len(sen_pair[1]) < 2 or len(sen_pair[1]) > 30:
                self._x_data.remove(sen_pair)
            elif (not re.findall("[0-9]",  "".join(sen_pair[0])) is None) or \
                    (not re.findall("[0-9]",  "".join(sen_pair[1])) is None):
                self._x_data.remove(sen_pair)
            elif (not re.findall("[a-zA-Z]",  "".join(sen_pair[0])) is None) or \
                    (not re.findall("[a-zA-Z]",  "".join(sen_pair[1])) is None):
                self._x_data.remove(sen_pair)
            elif (sen_pair[0] in self._remove_sign) or (sen_pair[1] in self._remove_sign):
                self._x_data.remove(sen_pair)

    def _get_sentences(self, batch_size):
        """ Get batch sentence from data

        :param batch_size:
        :return:
        """
        return [self._x_data[np.random.randint(0, len(self._x_data) - 1)] for _ in range(batch_size)]

    def _padding_zero(self, sen, max_sequence_length=None):
        """ Get each batch samples

        :param sen: inputs sentence
        :param max_sequence_length:
        :return: encoder_inputs, decoder_inputs, decoder_targets,
                  Type: np.array
                  Shape: (batch_size, time_steps, embed_size)
        """
        max_sequence_x_length, max_sequence_y_length = None, None

        input_x = [[self._vocab_dict[x]
                    if self._vocab_dict.get(x) else self._vocab_dict["_UNK_"] for x in s[0]] for s in sen]

        input_y = [[self._vocab_dict["_GO_"]] + (sequence)
                   for sequence in
                   [[self._vocab_dict[x] if self._vocab_dict.get(x) else self._vocab_dict["_UNK_"]
                     for x in s[1]] for s in sen]]

        target_y = [(sequence) + [self._vocab_dict["_EOS_"]]
                    for sequence in
                    [[self._vocab_dict[x] if self._vocab_dict.get(x) else self._vocab_dict["_UNK_"]
                      for x in s[1]] for s in sen]]

        sequence_x_lengths = [len(seq) for seq in input_x]
        if max_sequence_x_length is None:
            max_sequence_x_length = max(sequence_x_lengths)

        sequence_y_lengths = [len(seq) for seq in input_y]
        if max_sequence_y_length is None:
            max_sequence_y_length = max(sequence_y_lengths)

        inputs_x_batch_major = np.zeros(shape=[len(input_x), max_sequence_x_length], dtype=np.int32)
        inputs_y_batch_major = np.zeros(shape=[len(input_y), max_sequence_y_length], dtype=np.int32)
        target_y_batch_major = np.zeros(shape=[len(target_y), max_sequence_y_length], dtype=np.int32)

        for i, seq in enumerate(input_x):
            for j, element in enumerate(seq):
                inputs_x_batch_major[i, j] = element

        for i, seq in enumerate(input_y):
            for j, element in enumerate(seq):
                inputs_y_batch_major[i, j] = element

        for i, seq in enumerate(target_y):
            for j, element in enumerate(seq):
                target_y_batch_major[i, j] = element

        x = inputs_x_batch_major
        y = inputs_y_batch_major
        y_ = target_y_batch_major

        return x, y, y_, sequence_x_lengths, sequence_y_lengths

    def get_batch(self, batch_size, max_sequence_length=None):
        inputs_sentence = self._get_sentences(batch_size)
        return self._padding_zero(inputs_sentence, max_sequence_length=max_sequence_length)

    @property
    def vocab_dict(self):
        return self._vocab_dict

    @property
    def r_vocab_dict(self):
        return self._r_vocab_dict


class RuleCorrection(object):

    def __init__(self):
        self._rule_path = "config/rule.json"
        self._rule_dict = {}
        self._load_rule()

    def _load_rule(self):
        with open(self._rule_path, "r", encoding="utf-8") as f:
            self._rule_dict = json.load(f)
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
