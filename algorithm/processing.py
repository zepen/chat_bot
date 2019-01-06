# -*- coding:utf-8 -*-
"""
数据预处理
"""
import json
from utils.loggings import log

# class ProcessingCorps(object):
#
#     def __init__(self):
#         x_data = [(corpus[2 * n], corpus[2 * n + 1]) for n in range(int(len(corpus) / 2))]
#
#     def train_sequences(self, vocab_size, batch_size):
#         while True:
#             yield [x_data[np.random.randint(0, len(x_data) - 1)] for _ in range(batch_size)]
#
#     def make_batch(sen, max_sequence_length=None):
#         max_sequence_x_length, max_sequence_y_length = None, None
#
#         input_x = [[vocab_dict[x] for x in s[0]] for s in sen]
#         input_y = [[vocab_dict["_GO_"]] + (sequence) for sequence in [[vocab_dict[x] for x in s[1]] for s in sen]]
#         target_y = [(sequence) + [vocab_dict["_EOS_"]] for sequence in [[vocab_dict[x] for x in s[1]] for s in sen]]
#
#         sequence_x_lengths = [len(seq) for seq in input_x]
#         if max_sequence_x_length is None:
#             max_sequence_x_length = max(sequence_x_lengths)
#
#         sequence_y_lengths = [len(seq) for seq in input_y]
#         if max_sequence_y_length is None:
#             max_sequence_y_length = max(sequence_y_lengths)
#
#         inputs_x_batch_major = np.zeros(shape=[len(input_x), max_sequence_x_length], dtype=np.int32)
#         inputs_y_batch_major = np.zeros(shape=[len(input_y), max_sequence_y_length], dtype=np.int32)
#         target_y_batch_major = np.zeros(shape=[len(target_y), max_sequence_y_length], dtype=np.int32)
#
#         for i, seq in enumerate(input_x):
#             for j, element in enumerate(seq):
#                 inputs_x_batch_major[i, j] = element
#
#         for i, seq in enumerate(input_y):
#             for j, element in enumerate(seq):
#                 inputs_y_batch_major[i, j] = element
#
#         for i, seq in enumerate(target_y):
#             for j, element in enumerate(seq):
#                 target_y_batch_major[i, j] = element
#
#         x = inputs_x_batch_major.swapaxes(0, 1)
#         y = inputs_y_batch_major.swapaxes(0, 1)
#         y_ = target_y_batch_major.swapaxes(0, 1)
#         return x, y, y_


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
