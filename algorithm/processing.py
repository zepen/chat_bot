# -*- coding:utf-8 -*-
"""
Processing Data
"""
import time
import numpy as np
from utils.load_files import LoadCorpus


class ProcessingCorps(object):

    def __init__(self, start_point=0, end_point=None):
        load_corpus = LoadCorpus()
        self._x_data = load_corpus.x_data[start_point:end_point]
        print("[INFO] Data size is {}.".format(len(self._x_data)))
        time.sleep(10)

    def _get_sentences(self, batch_size):
        """ Get batch sentence from data

        :param batch_size:
        :return:
        """
        return [self._x_data[np.random.randint(0, len(self._x_data) - 1)] for _ in range(batch_size)]

    def _padding_zero(self, sen):
        """ Get each batch samples

        :param sen: inputs sentence
        :return: encoder_inputs, decoder_inputs, decoder_targets,
                  Type: np.array
                  Shape: (batch_size, time_steps, embed_size)
        """
        max_sequence_x_length, max_sequence_y_length = None, None

        input_x = [[x for x in s[0]] for s in sen]

        input_y = [["_GO_"] + sequence for sequence in [[y for y in s[1]] for s in sen]]

        target_y = [sequence + ["_EOS_"] for sequence in [[x for x in s[1]] for s in sen]]

        sequence_x_lengths = [len(seq) for seq in input_x]
        if max_sequence_x_length is None:
            max_sequence_x_length = max(sequence_x_lengths)

        sequence_y_lengths = [len(seq) for seq in input_y]
        if max_sequence_y_length is None:
            max_sequence_y_length = max(sequence_y_lengths)

        # noinspection PyTypeChecker
        inputs_x_batch_major = np.full((len(input_x), max_sequence_x_length), "_PAD_",  dtype='U5')
        # noinspection PyTypeChecker
        inputs_y_batch_major = np.full((len(input_y), max_sequence_y_length), "_PAD_", dtype='U5')
        # noinspection PyTypeChecker
        target_y_batch_major = np.full((len(target_y), max_sequence_y_length), "_PAD_", dtype='U5')

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

    def get_batch(self, batch_size):
        inputs_sentence = self._get_sentences(batch_size)
        return self._padding_zero(inputs_sentence)
