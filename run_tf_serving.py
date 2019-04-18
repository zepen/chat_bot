# -*- coding:utf-8 -*-
"""
启动tf_serving
"""
import os
import argparse

parser = argparse.ArgumentParser(description='This script argument')
parser.add_argument('--char_dict', help='')
parser.add_argument('--model_version', default="001")
args = parser.parse_args()

DOCKER_BUILD = "docker build -t tensorflow/serving ."
DOCKER_RUN = "docker run -t --rm " +  "-p 8501:8501 --privileged=true "
DOCKER_FILE_PATH = "docker/"
PB_MODEL_PATH = "/model/" + args.model_version


def docker_build():
    """ 构建docker镜像
    :return:
    """
    os.chdir(DOCKER_FILE_PATH)
    os.system(DOCKER_BUILD)


def docker_run():
    """ 启动docker容器
    :return:
    """
    global DOCKER_RUN
    os.chdir("..")
    path = os.getcwd().replace("\\", "/")
    v = '-v  "' + path + PB_MODEL_PATH + ':/models/chat_bot" '
    e = "-e MODEL_NAME=chat_bot tensorflow/serving "
    DOCKER_RUN += (v + e)
    print(DOCKER_RUN)
    os.system(DOCKER_RUN)


if __name__ == '__main__':
    docker_build()
    docker_run()
