# -*- coding: utf-8 -*-
"""
测试tf_serving是否通畅
"""
import os
from config import ModelConfig
from utils.connect import ConnectionTFServing
from utils.load_files import LoadDictionary
from algorithm.seq2seq import Seq2SeqModel
from algorithm.processing import RuleCorrection

os.chdir("..")

con_tf_s = ConnectionTFServing()
load_dict = LoadDictionary()
m_config = ModelConfig()
rule_c = RuleCorrection()
vd = load_dict.vocab_dict
rvd = load_dict.r_vocab_dict
print("URL {}".format(con_tf_s.url))


def test_get_tf_serving():
    print(Seq2SeqModel.predict_func(
        "傻逼", vd, rvd, con_tf_s, m_config, rule_c)
    )


if __name__ == '__main__':
    test_get_tf_serving()
