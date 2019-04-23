# -*- coding:utf-8 -*-
"""
启动tf_serving
"""
from serve.tf_serving import docker_run

if __name__ == '__main__':
    docker_run("docker/")
