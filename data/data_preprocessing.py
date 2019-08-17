# -*- coding:utf-8 -*-
"""
数据预处理，筛选语料
"""
import os
import time
from algorithm.language_model import BaiduDnnLM

min_len = 5
max_len = 30
max_ppl = 1200
baidu_lm = BaiduDnnLM()
source_corpus_path = os.path.dirname(__file__) + "/source_corpus.txt"
train_corpus_path = os.path.dirname(__file__) + "/train_corpus.txt"
# match_chinese = re.compile("[\u4e00-\u9fa5]+")
# match_english = re.compile("[a-zA-Z]+")


def pre_processing_func(line):
    global new_no
    read_line = line.split("\t")
    sentence_a = read_line[1].split("|")
    sentence_b = read_line[2].split("|")
    try:
        if (min_len < len(sentence_a) < max_len) and (min_len < len(sentence_b) < max_len):
            print("".join(sentence_a) + "|" + "".join(sentence_b))
            sentence_a_ppl = baidu_lm.get_ppl(text="".join(sentence_a))
            sentence_b_ppl = baidu_lm.get_ppl(text="".join(sentence_b))
            if (sentence_a_ppl < max_ppl) and (sentence_b_ppl < max_ppl):
                with open(train_corpus_path, "a+", encoding="utf-8") as f_out:
                    f_out.write(str(new_no) + "\t" + read_line[1] + "\t" + read_line[2])
                print("[INFO] read_line No.{}".format(read_line[0]))
                new_no += 1
    except Exception as e:
        print("[ERROR] {}, read_line is {}".format(str(e), read_line))


if __name__ == '__main__':
    with open(source_corpus_path, "r", encoding="utf-8") as f:
        source_corpus_list = f.readlines()
    print("Source_corpus len: {}".format(len(source_corpus_list)))
    start_time = time.time()
    print("Start transform...")
    new_no = 0
    for corpus in source_corpus_list:
        pre_processing_func(corpus)
    end_time = time.time()
    print("Cost time: {} seconds".format(end_time - start_time))
