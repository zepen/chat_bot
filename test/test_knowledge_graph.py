# -*- coding: utf-8 -*-
"""
测试neo4j图数据库
"""
import os
from algorithm.knowledge_graph import KnowledgeGraph

os.chdir("..")


def test_knowledge_graph():
    kg = KnowledgeGraph()


if __name__ == '__main__':
    test_knowledge_graph()