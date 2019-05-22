# -*- coding: utf-8 -*-
"""
测试ltp模型
"""
import os
from algorithm.named_entity_recognizer import Ltp

os.chdir("..")


def test_ltp():
    ltp = Ltp()
    print(ltp.get_entity("我喜欢玩王者荣耀!"))
