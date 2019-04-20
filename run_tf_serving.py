# -*- coding:utf-8 -*-
"""
启动tf_serving
"""
import os
import argparse

parser = argparse.ArgumentParser(description='This script argument')
parser.add_argument('--name', default='chat_bot_serve')
parser.add_argument('--port', default=8501)
args = parser.parse_args()

DOCKER_BUILD = "docker build -t tensorflow/serving ."
DOCKER_RUN = "docker run -t --rm -p " + \
             str(args.port) + ":" + str(args.port) + \
             " --name=" + args.name + \
             " --privileged=true "
DOCKER_FILE_PATH = "docker/"


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
    v = '-v  "' + path + '/model/:/models/chat_bot" '
    e = "-e MODEL_NAME=chat_bot tensorflow/serving "
    DOCKER_RUN += (v + e)
    print(DOCKER_RUN)
    os.system(DOCKER_RUN)


if __name__ == '__main__':
    docker_build()
    docker_run()
