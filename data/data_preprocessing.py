# -*- coding:utf-8 -*-
"""
数据预处理，筛选语料
"""
import re

min_len = 5
max_len = 30
match_chinese = re.compile("[\u4e00-\u9fa5]+")
match_english = re.compile("[a-zA-Z]+")


def pre_processing_func():
    with open("source_corpus.txt", "r", encoding="utf-8") as f_in, \
            open("train_corpus.txt", "w", encoding="utf-8") as f_out:
        new_no = 0
        for line in f_in:
            read_line = line.split("\t")
            sentence_a = read_line[1].split("|")
            sentence_b = read_line[2].split("|")
            if (min_len < len(sentence_a) < max_len) and (min_len < len(sentence_b) < max_len):
                f_out.write(str(new_no) + "\t" + read_line[1] + "\t" + read_line[2])
                new_no += 1


if __name__ == '__main__':
    pre_processing_func()
