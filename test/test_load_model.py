# -*- coding: utf-8 -*-
"""
测试加载模型文件
"""
import os
import tensorflow as tf
from algorithm.seq2seq import Seq2Seq
from joblib import load
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops

os.chdir("..")

# 加载训练好模型的超参数
with open("logs/hyper_parameters.pkl", "rb") as f:
    hp = load(f)

hp.device = "cpu"
hp.gpu_no = "0"
hp.layer_num = 3
hp.beam_search = 1
hp.beam_size = 5
hp.mode = "predict"
hp.encoder_keep_prob = 1.0
hp.decoder_keep_prob = 1.0
hp.max_decode_len = 100


def test_load_ckpt_model():
    seq2seq_predict = Seq2Seq(hp=hp)
    with tf.Session() as sess:
        tf.tables_initializer().run()
        seq2seq_predict.load_model(sess)
        # for tensor in tf.get_default_graph().as_graph_def().node:
        #     print(tensor.name)
        print(sess.graph.get_tensor_by_name("inputs/inputs_sentence:0"))
        print(sess.graph.get_tensor_by_name("inputs/encoder_inputs_length:0"))
        print(sess.graph.get_tensor_by_name("inputs/batch_size:0"))
        print(sess.graph.get_tensor_by_name("predict/decoder/transpose_1:0"))
        print(sess.graph.get_tensor_by_name("predict/prediction/index_to_string_Lookup:0"))

        print("Look_ids: {}".format(
            sess.run("inputs/predict_encoder_inputs:0", feed_dict={
                "inputs/inputs_sentence:0": [list("我还喜欢她,怎么办")]
            })
        ))
        print("Embeddings_vector: {}".format(
            sess.run("encoder/embedding/embedding_lookup:0", feed_dict={
                "inputs/inputs_sentence:0": [list("今天天气很好!")]
            })
        ))
        print("Predict_sequence: {}".format(
            sess.run("predict/decoder/transpose_1:0", feed_dict={
                "inputs/inputs_sentence:0": [list("今天天气很好!")],
                "inputs/encoder_inputs_length:0": [7],
                "inputs/batch_size:0": [1]
            })
        ))
        sentence = sess.run(
            "predict/prediction/index_to_string_Lookup:0", feed_dict={
                "inputs/inputs_sentence:0": [list("想不想吃肯德基？")],
                "inputs/encoder_inputs_length:0": [7],
                "inputs/batch_size:0": [1]}
        )
        print("Decode Result: {}".format("".join([w.decode("utf-8") for w in sentence])))


def test_load_pb_model():
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],  "model/001/")
        # for tensor in tf.get_default_graph().as_graph_def().node:
        #     print(tensor.name)
        print(sess.graph.get_tensor_by_name("inputs/encoder_inputs:0"))
        print(sess.graph.get_tensor_by_name("inputs/encoder_inputs_length:0"))
        print(sess.graph.get_tensor_by_name("inputs/batch_size:0"))
        print(sess.graph.get_tensor_by_name("predict/decoder/transpose_1:0"))
        print(sess.graph.get_tensor_by_name("predict/prediction/index_to_string_Lookup:0"))
        print("Look_ids: {}".format(
            sess.run("inputs/predict_encoder_inputs:0", feed_dict={
                "inputs/inputs_sentence:0": [list("今天天气很好!")]
            })
        ))
        print("Predict_sequence: {}".format(
            sess.run("predict/decoder/transpose_1:0", feed_dict={
                "inputs/inputs_sentence:0": [list("今天天气很好!")],
                "inputs/encoder_inputs_length:0": [7],
                "inputs/batch_size:0": [1]
            })
        ))
        sentence = sess.run(
            "predict/prediction/index_to_string_Lookup:0", feed_dict={
                "inputs/inputs_sentence:0": [list("我还喜欢她,怎么办")],
                "inputs/encoder_inputs_length:0": [7],
                "inputs/batch_size:0": [1]}
        )
        print("Decode Result: {}".format("".join([w.decode("utf-8") for w in sentence])))
