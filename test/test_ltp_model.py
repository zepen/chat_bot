# -*- coding: utf-8 -*-
"""
测试ltp模型
"""
import os
from algorithm.named_entity_recognizer import Ltp

os.chdir("..")


def test_ltp():
    ltp = Ltp()
    print(ltp.get_entity("姚明是谁？"))
    print(ltp.get_entity("郭敬明是谁？"))
    print(ltp.get_entity("任正非是谁？"))
    print(ltp.get_entity("宋江是谁？"))
    print(ltp.get_entity("孙悟空是谁？"))
    print(ltp.get_entity("特朗普是谁？"))
    print(ltp.get_entity("勒布朗詹姆斯是谁？"))
    print(ltp.get_entity("你认识姚明么"))
