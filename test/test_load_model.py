# -*- coding: utf-8 -*-
"""
测试加载模型
"""
import os
import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops

os.chdir("..")


def test_load_ckpt_model():
    saver = tf.train.import_meta_graph("logs/s2s.ckpt-0.meta")
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint("./logs"))
        for tensor in tf.get_default_graph().as_graph_def().node:
            print(tensor.name)
        print(sess.graph.get_tensor_by_name("inputs/encoder_inputs:0"))
        print(sess.graph.get_tensor_by_name("inputs/encoder_inputs_length:0"))
        print(sess.graph.get_tensor_by_name("inputs/batch_size:0"))
        print(sess.graph.get_tensor_by_name("predict/decoder/transpose_1:0"))
        print("Decode Result: {}".format(
            sess.run("predict/decoder/transpose_1:0", feed_dict={
                "inputs/encoder_inputs:0": [[1, 2, 3, 0]],
                "inputs/encoder_inputs_length:0": [4],
                "inputs/batch_size:0": [1]
            })
        ))


def test_load_pb_model():
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],  "model/004/")
        for tensor in tf.get_default_graph().as_graph_def().node:
            print(tensor.name)
        print(sess.graph.get_tensor_by_name("inputs/encoder_inputs:0"))
        print(sess.graph.get_tensor_by_name("inputs/encoder_inputs_length:0"))
        print(sess.graph.get_tensor_by_name("inputs/batch_size:0"))
        print(sess.graph.get_tensor_by_name("predict/decoder/transpose_3:0"))
        print("Decode Result: {}".format(
            sess.run("predict/decoder/transpose_3:0", feed_dict={
                "inputs/encoder_inputs:0": [[1, 2, 3, 0]],
                "inputs/encoder_inputs_length:0": [4],
                "inputs/batch_size:0": [1]
            })
        ))
