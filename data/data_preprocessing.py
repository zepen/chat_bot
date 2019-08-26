# -*- coding:utf-8 -*-
"""
数据预处理，筛选语料
"""
import os
import time
import re
from algorithm.language_model import BaiduDnnLM

min_len = 5
max_len = 30
max_ppl = 1200
baidu_lm = BaiduDnnLM()
source_corpus_path = os.path.dirname(__file__) + "/source_corpus.txt"
train_corpus_path = os.path.dirname(__file__) + "/train_corpus.txt"


def regular(sen):
    """ 句子规范化，统一语料标点符号

    :param sen: 输入句子
    :return:
    """
    sen = sen.replace('/', '')
    sen = re.sub(r'…{1,100}', '…', sen)
    sen = re.sub(r'\.{3,100}', '…', sen)
    sen = re.sub(r'···{2,100}', '…', sen)
    sen = re.sub(r',{1,100}', '，', sen)
    sen = re.sub(r'\.{1,100}', '。', sen)
    sen = re.sub(r'。{1,100}', '。', sen)
    sen = re.sub(r'\?{1,100}', '？', sen)
    sen = re.sub(r'？{1,100}', '？', sen)
    sen = re.sub(r'!{1,100}', '！', sen)
    sen = re.sub(r'！{1,100}', '！', sen)
    sen = re.sub(r'~{1,100}', '～', sen)
    sen = re.sub(r'～{1,100}', '～', sen)
    sen = re.sub(r'[“”]{1,100}', '"', sen)
    sen = re.sub('[^\w\u4e00-\u9fff"。，？！～·]+', '', sen)
    sen = re.sub(r'[ˇˊˋˍεπのゞェーω]', '', sen)
    return sen


def good_line(line):
    """ 判断一句话是否是好的语料,即判断一段文本当中文，字母，数字，特殊符号所占比例，中文比例越高，认为该语料越好
    :param line: 输入文段
    :return: bool
    """
    if len(line) == 0:
        return False
    ch_count = 0
    for c in line:
        # 中文字符范围
        if '\u4e00' <= c <= '\u9fff':
            ch_count += 1
    if ch_count / float(len(line)) >= 0.8 and len(re.findall(r'[a-zA-Z0-9]', ''.join(line))) < 3 \
            and len(re.findall(r'[ˇˊˋˍεπのゞェーω]', ''.join(line))) < 3:
        return True
    return False


def ppl_line(sentence_a, sentence_b):
    """  调用百度语言模型API，剔除ppl较高语料
    :param sentence_a: 对话上句
    :param sentence_b: 对话下句
    :return:
    """
    sentence_a_ppl = baidu_lm.get_ppl(text="".join(sentence_a))
    sentence_b_ppl = baidu_lm.get_ppl(text="".join(sentence_b))
    if (sentence_a_ppl < max_ppl) and (sentence_b_ppl < max_ppl):
        return True
    else:
        return False


def pre_processing_func(line):
    global new_no
    read_line = line.split("\t")
    sentence_a = read_line[1]
    sentence_b = read_line[2]
    try:
        if (min_len < len(sentence_a) < max_len) and (min_len < len(sentence_b) < max_len):
            print(sentence_a + "|" + sentence_b)
            sentence_a = regular(sentence_a)
            sentence_b = regular(sentence_b)
            if good_line(sentence_a) and good_line(sentence_b):
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
    f_out = open(train_corpus_path, "a+", encoding="utf-8")
    for corpus in source_corpus_list:
        pre_processing_func(corpus)
    f_out.close()
    end_time = time.time()
    print("Cost time: {} seconds".format(end_time - start_time))
