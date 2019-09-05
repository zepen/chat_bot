# -*- coding:utf-8 -*-
"""
数据预处理，入模语料
"""
import os
import time

msr_corpus_path = os.path.dirname(__file__) + "/msr_corpus.txt"
msr_train_path = os.path.dirname(__file__) + "/msr_train.txt"


def pre_processing_func(line_):
    """  预处理函数

    :param line_: 预处理行文本
    :return:
    """
    read_line = line_.split("\t")
    no = read_line[0]
    text = read_line[1].replace("\n", "")
    corpus = "|".join([t.split("/")[0] for t in text.split(" ") if len(t)])
    label = "|".join([t.split("/")[1] for t in text.split(" ") if len(t)])
    f_out.write(no + "\t" + corpus + "\t" + label + "\n")


if __name__ == '__main__':
    with open(msr_corpus_path, "r", encoding="utf-8") as f:
        mrs_corpus_list = f.readlines()
    print("msr_corpus len: {}".format(len(mrs_corpus_list)))
    start_time = time.time()
    print("Start transform...")
    new_no = 0
    f_out = open(msr_train_path, "a+", encoding="utf-8")
    for line in mrs_corpus_list:
        pre_processing_func(line)
    f_out.close()
    end_time = time.time()
    print("Cost time: {} seconds".format(end_time - start_time))
