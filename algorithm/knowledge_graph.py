# -*- coding:utf-8 -*-
"""
定义KG，用于关系查询
"""
import py2neo
import pandas as pd
from utils.connect import ConnectionNeo4j
from py2neo import Graph, Node, Relationship


class KnowledgeGraph(object):

    def __init__(self):
        conn_neo4j = ConnectionNeo4j()
        self._graph = Graph(host=conn_neo4j.ip, auth=(conn_neo4j.username, conn_neo4j.password))

    def __repr__(self):
        self._object = "[INFO] The neo4j version is {}.".format(py2neo.__version__)

    def load_file(self, cypher):
        self._graph.run(cypher)

    def add_node(self, labels, **kwargs):
        node = Node(labels, **kwargs)
        self._graph.create(node)

    def delete_node(self):
        self._graph.delete_all()

    # def find(self, label):
    #     return self._graph.find_one(label=label)

    def find(self):
        data = self._graph.data('MATCH (p:FUCK) return p')
        df = pd.DataFrame(data)
        print(df)

    def match(self):
        pass
