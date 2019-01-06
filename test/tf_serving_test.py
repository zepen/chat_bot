# -*- coding: utf-8 -*-
"""
测试tf_serving是否通畅
"""
import os
from utils.connect import ConnectionTFServing
from utils.load_files import LoadDictionary
from algorithm.seq2seq import Seq2SeqModel
from utils.model_config import ModelConfig

os.chdir("..")


con_tf_s = ConnectionTFServing()
load_dict = LoadDictionary()
m_config = ModelConfig()
vd = load_dict.vocab_dict
rvd = load_dict.r_vocab_dict

print("URL {}".format(con_tf_s.url))


if __name__ == '__main__':
    print(Seq2SeqModel.predict_fun("傻逼", vd, rvd, con_tf_s, m_config))
