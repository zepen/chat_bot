# -*- coding: utf-8 -*-
"""
测试加载模型
"""
import os
import tensorflow as tf

os.chdir("..")


def test_load_model():
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],  "model/001/")
        for tensor in tf.get_default_graph().as_graph_def().node:
            print(tensor.name)
        print(sess.graph.get_tensor_by_name("inputs/encoder_inputs:0"))
        print(sess.graph.get_tensor_by_name("inputs/encoder_inputs_length:0"))
        print(sess.graph.get_tensor_by_name("inputs/batch_size:0"))
        print("Decode Result: {}".format(
            sess.run("predict/decoder/transpose_1:0", feed_dict={
                "inputs/encoder_inputs:0": [[1, 2, 3, 0]],
                "inputs/encoder_inputs_length:0": [4],
                "inputs/batch_size:0": [1]
            })))
