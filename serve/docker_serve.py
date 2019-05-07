# -*- coding:utf-8 -*-
"""
Docker serve
"""
import os
import json
import requests as req
from utils.loggings import log
from utils.connect import ConnectionTFServing, ConnectionNeo4j

conn_tf_serving = ConnectionTFServing()
conn_neo4j = ConnectionNeo4j()

DOCKER_BUILD_TFS = "docker build -t tensorflow/serving --target tensorflow_serving ."
DOCKER_BUILD_NEO4J = "docker build -t neo4j --target neo4j_serving ."
DOCKER_RUN_TFS = "docker run -d -p " + str(conn_tf_serving.port) + ":" + str(conn_tf_serving.port) + \
                 " --name=" + conn_tf_serving.name + " --privileged=true "
DOCKER_RUN_NEO4J = "docker run -d -p " + str(conn_neo4j.port_1) + ":" + str(conn_neo4j.port_1) + " " + \
                   "-p " + str(conn_neo4j.port_2) + ":" + str(conn_neo4j.port_2) + " " + \
                   "-p " + str(conn_neo4j.port_3) + ":" + str(conn_neo4j.port_3) + " " + \
                   "--name=" + conn_neo4j.name + " " + \
                   "--env=NEO4J_AUTH=" + str(conn_neo4j.username) + "/" + str(conn_neo4j.password) + " "


def docker_build(docker_file_path):
    """ 构建docker镜像
    :param: docker_file_path
    :return:
    """
    os.chdir(docker_file_path)
    os.system(DOCKER_BUILD_TFS)
    os.system(DOCKER_BUILD_NEO4J)


def docker_run(docker_file_path):
    """ 启动docker容器
    :param: docker_file_path
    :return:
    """
    global DOCKER_RUN_TFS, DOCKER_RUN_NEO4J
    docker_build(docker_file_path)
    os.chdir("..")
    path = os.getcwd().replace("\\", "/").replace(":", "").lower()
    v_1 = "-v /" + path + "/model/:/models/chat_bot "
    v_2 = "-v /" + path + "/data/:/var/lib/neo4j/import"
    e = "-e MODEL_NAME=chat_bot"
    DOCKER_RUN_TFS += (v_1 + e + " tensorflow/serving")
    DOCKER_RUN_NEO4J += (v_2 + " neo4j:latest")
    print("[tensorflow-serving docker run command]: {}".format(DOCKER_RUN_TFS))
    print("[neo4j docker run command]: {}".format(DOCKER_RUN_NEO4J))
    os.system(DOCKER_RUN_TFS)
    os.system(DOCKER_RUN_NEO4J)


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