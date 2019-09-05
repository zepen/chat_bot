# -*- coding:utf-8 -*-
"""
定义DateSet, 用于从文件读取数据
"""
import tensorflow as tf


class BaseDataSet(object):

    def __init__(self):
        self.data_path = "./data/"
        with tf.gfile.GFile(self.data_path + "vocab_dict.txt") as f:
            self._vocab_size = len(f.readlines())
        self._string2index_table = tf.contrib.lookup.string_to_index_table_from_file(
            vocabulary_file=self.data_path + "vocab_dict.txt", num_oov_buckets=0, default_value=1)
        self._index2string_table = tf.contrib.lookup.index_to_string_table_from_file(
            vocabulary_file=self.data_path + "vocab_dict.txt")
        self._iterator = None

    def __repr__(self):
        return "<This is BaseDataSet>"

    def _process_func(self, string):
        """  预处理函数
        """
        pass

    def iterator_func(self, epoch, batch_size_):
        """   获取训练批次
        """
        pass

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def string2index_table(self):
        return self._string2index_table

    @property
    def index2string_table(self):
        return self._index2string_table

    @property
    def iterator(self):
        return self._iterator
