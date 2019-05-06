# -*- coding:utf-8 -*-
"""
定义KG，用于关系查询
"""
import py2neo
from utils.connect import ConnectionNeo4j
from py2neo import Graph, Node, Relationship


class KnowledgeGraph(object):

    def __init__(self, file_name):
        conn_neo4j = ConnectionNeo4j()
        self._graph = Graph(host=conn_neo4j.ip, auth=(conn_neo4j.username, conn_neo4j.password))
        self._graph.run(
            'LOAD CSV FROM "file:///' + file_name + '" AS line '
            'CREATE (:Artist { name: line[1], year: toInteger(line[2])})'
        )

    def __str__(self):
        self._object = "[INFO] The neo4j version is {}.".format(py2neo.__version__)

    def add_node(self, labels, **kwargs):
        node = Node(labels, **kwargs)
        self._graph.create(node)

    def delete_node(self):
        self._graph.delete_all()
