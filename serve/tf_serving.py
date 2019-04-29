# -*- coding:utf-8 -*-
"""
TF_SERVING RESTFUL API
"""
import os
import json
import requests as req
from utils.loggings import log
from utils.connect import ConnectionTFServing

conn_tf_serving = ConnectionTFServing()

DOCKER_BUILD = "docker build -t tensorflow/serving ."
DOCKER_RUN = "docker run -t --rm -p " + str(conn_tf_serving.port) + ":" + str(conn_tf_serving.port) + \
             " --name=" + conn_tf_serving.name + " --privileged=true "


def docker_build(docker_file_path):
    """ 构建docker镜像
    :param: docker_file_path
    :return:
    """
    os.chdir(docker_file_path)
    os.system(DOCKER_BUILD)


def docker_run(docker_file_path):
    """ 启动docker容器
    :param: docker_file_path
    :return:
    """
    global DOCKER_RUN
    docker_build(docker_file_path)
    os.chdir("..")
    path = os.getcwd().replace("\\", "/").replace(":", "").lower()
    v = "-v /home/share:/models/chat_bot "
    e = "-e MODEL_NAME=chat_bot tensorflow/serving &"
    DOCKER_RUN += (v + e)
    print(DOCKER_RUN)
    os.system(DOCKER_RUN)


def get_predict_result(x):
    """
    :param x: 待预测样本
    :return: dict 预测结果
    """
    assert isinstance(x, dict)
    try:
        r = req.post(url=conn_tf_serving.url, data=json.dumps(x))
        log.info("[INFO] " + str(r.status_code))
        return r.json()
    except Exception as e:
        log.error("[ERROR] " + str(e))


def predict_func(input_text, vocab_dict, r_vocab_dict, m_config, rule_correction):
    """  解码预测

    :param input_text 输出文本
    :param vocab_dict: 词表
    :param r_vocab_dict: 反转词表
    :param m_config 配置对象
    :param rule_correction 规则修正对象
    :return: str
    """
    x = [vocab_dict[x] if vocab_dict.get(x) else vocab_dict["_UNK_"] for x in list(input_text)]
    data = {
        "instances": [
            {
                "encoder_inputs": x,
                "encoder_inputs_length": len(x),
                "batch_size": 1
            },
        ]
    }
    predict_res = get_predict_result(data)["predictions"][0]
    output_text = rule_correction(
        "".join([r_vocab_dict[y] for y in predict_res if r_vocab_dict[y] != "_EOS_"])
    )
    return m_config.replace_sentence if output_text is None else output_text
