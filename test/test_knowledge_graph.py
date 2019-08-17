# -*- coding: utf-8 -*-
"""
测试neo4j图数据库
"""
from algorithm.knowledge_graph import KnowledgeGraph


def test_knowledge_graph():
    kg = KnowledgeGraph()
    # kg.delete_node()
    # kg.add_node("FUCK")
    kg.delete_node()
    kg.load_file(
        'LOAD CSV WITH HEADERS  FROM "file:///genre.csv" AS line '
        'MERGE (p:Genre{gid:toInteger(line.gid),name:line.gname})'
    )
    kg.load_file(
        'LOAD CSV WITH HEADERS FROM "file:///person.csv" AS line '
        'MERGE (p:Person { pid:toInteger(line.pid),birth:line.birth,'
        'death:line.death,name:line.name, biography:line.biography, birthplace:line.birthplace})'
    )
    kg.load_file(
        'LOAD CSV WITH HEADERS  FROM "file:///movie.csv" AS line '  
        'MERGE (p:Movie{mid:toInteger(line.mid),title:line.title,introduction:line.introduction, '
        'rating:toFloat(line.rating),releasedate:line.releasedate})'
    )
    kg.load_file(
        'LOAD CSV WITH HEADERS FROM "file:///person_to_movie.csv" AS line '
        'match (from:Person{pid:toInteger(line.pid)}),(to:Movie{mid:toInteger(line.mid)})  '
        'merge (from)-[r:actedin{pid:toInteger(line.pid),mid:toInteger(line.mid)}]->(to)'
    )

    kg.load_file(
        'LOAD CSV WITH HEADERS FROM "file:///movie_to_genre.csv" AS line '
        'match (from:Movie{mid:toInteger(line.mid)}),(to:Genre{gid:toInteger(line.gid)})  '
        'merge (from)-[r:is{mid:toInteger(line.mid),gid:toInteger(line.gid)}]->(to)'
    )
