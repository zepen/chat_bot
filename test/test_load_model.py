# -*- coding: utf-8 -*-
"""
测试加载模型文件
"""
import os
import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops

os.chdir("..")


def test_load_ckpt_model():
    saver = tf.train.import_meta_graph("logs/s2s.ckpt.meta")
    with tf.Session() as sess:
        tf.tables_initializer().run()
        saver.restore(sess, tf.train.latest_checkpoint("logs/"))
        for tensor in tf.get_default_graph().as_graph_def().node:
            print(tensor.name)
        print(sess.graph.get_tensor_by_name("inputs/encoder_inputs:0"))
        print(sess.graph.get_tensor_by_name("inputs/encoder_inputs_length:0"))
        print(sess.graph.get_tensor_by_name("inputs/batch_size:0"))
        print(sess.graph.get_tensor_by_name("predict/decoder/transpose_1:0"))
        print(sess.graph.get_tensor_by_name("predict/prediction/index_to_string_Lookup:0"))
        print("Look_ids: {}".format(
            sess.run("map_sequence/hash_table_Lookup:0", feed_dict={
                "inputs/encoder_inputs:0": [list("今天天气很好!")]
            })
        ))
        print("Embeddings_vector: {}".format(
            sess.run("embedding/embedding_lookup:0", feed_dict={
                "inputs/encoder_inputs:0": [list("今天天气很好!")]
            })
        ))
        print("Predict_sequence: {}".format(
            sess.run("predict/decoder/transpose_1:0", feed_dict={
                "inputs/encoder_inputs:0": [list("今天天气很好!")],
                "inputs/encoder_inputs_length:0": [7],
                "inputs/batch_size:0": [1]
            })
        ))
        print("Decode Result: {}".format(
            [w.decode("utf-8") for w in
             sess.run("predict/prediction/index_to_string_Lookup:0", feed_dict={
                "inputs/encoder_inputs:0": [list("今天天气很好!")],
                "inputs/encoder_inputs_length:0": [7],
                "inputs/batch_size:0": [1]
            })[0]]
        ))


def test_load_pb_model():
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],  "model/001/")
        for tensor in tf.get_default_graph().as_graph_def().node:
            print(tensor.name)
        print(sess.graph.get_tensor_by_name("inputs/encoder_inputs:0"))
        print(sess.graph.get_tensor_by_name("inputs/encoder_inputs_length:0"))
        print(sess.graph.get_tensor_by_name("inputs/batch_size:0"))
        print(sess.graph.get_tensor_by_name("predict/decoder/transpose_1:0"))
        print(sess.graph.get_tensor_by_name("predict/prediction/index_to_string_Lookup:0"))
        print("Look_ids: {}".format(
            sess.run("map_sequence/hash_table_Lookup:0", feed_dict={
                "inputs/encoder_inputs:0": [list("今天天气很好!")]
            })
        ))
        print("Predict_sequence: {}".format(
            sess.run("predict/decoder/transpose_1:0", feed_dict={
                "inputs/encoder_inputs:0": [list("今天天气很好!")],
                "inputs/encoder_inputs_length:0": [7],
                "inputs/batch_size:0": [1]
            })
        ))
        print("Decode Result: {}".format(
            [w.decode("utf-8") for w in
             sess.run("predict/prediction/index_to_string_Lookup:0", feed_dict={
                "inputs/encoder_inputs:0": [list("今天天气很好!")],
                "inputs/encoder_inputs_length:0": [7],
                "inputs/batch_size:0": [1]
             })[0]]
        ))
