# -*- coding:utf-8 -*-
"""
定义seq2seq模型
"""
import numpy as np
import tensorflow as tf


class Seq2SeqModel(object):

    def __init__(self):
        pass
        # tf.reset_default_graph()
        # gpu_options = tf.GPUOptions(allow_growth=True)
        # config = tf.ConfigProto(
        #     device_count={"CPU": 8},
        #     log_device_placement=True,
        #     gpu_options=gpu_options
        # )

    def _input_layer(self):
        with tf.device('/cpu:0'):
            with tf.name_scope("inputs"):
                self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
                self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
                self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

    def _embedding_layer(self):
        with tf.device('/cpu:0'):
            with tf.name_scope("embeddings"):
                embeddings = tf.Variable(
                    tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
                encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
                decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

    def _encode_layer(self):
        with tf.device('/cpu:0'):
            with tf.name_scope("encoder"):
                encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
                encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                    encoder_cell, encoder_inputs_embedded,
                    dtype=tf.float32, time_major=True,
                )

    def _decode_layer(self):
        with tf.device('/cpu:0'):
            with tf.name_scope("decoder"):
                decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
                decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
                    decoder_cell, decoder_inputs_embedded,
                    initial_state=encoder_final_state,
                    dtype=tf.float32, time_major=True, scope="plain_decoder",
                )

    def _output_layer(self):
        with tf.device('/cpu:0'):
            with tf.name_scope("full_connect"):
                decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
            with tf.name_scope("softmax"):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
                    logits=decoder_logits,
                )
    def _predict_layer(self):
        with tf.device('/cpu:0'):
            with tf.name_scope("predict"):
                decoder_prediction = tf.argmax(decoder_logits, 2)

    def _loss_fun(self):
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(cross_entropy)

    def _train_op(self):
        with tf.name_scope("trian_op"):
            train_op = tf.train.AdamOptimizer().minimize(loss)

    @staticmethod
    def predict_fun(input_text, vd, rvd, con_tf_s):
        """
        :param input_text 输出文本
        :param vd: 词表
        :param rvd: 反转词表
        :param con_tf_s 连接tensorflow serving object
        :return: str
        """
        data = {
            "instances": [
                {
                    "encoder_inputs": [],
                    "decoder_inputs": []
                },
            ]
        }
        x = [vd[x] if vd.get(x) else vd["_UNK_"] for x in list(input_text)]
        data["instances"][0]["encoder_inputs"].extend(x)
        data["instances"][0]["decoder_inputs"].extend([vd["_GO_"]])
        res = []
        while 1:
            con_tf_s.calculate_predict_result(data)
            predict_res = con_tf_s.predict_result["predictions"][0][-1]
            if rvd[predict_res] == "_EOS_":
                break
            res.append(predict_res)
            data["instances"][0]["decoder_inputs"].append(predict_res)
        return "".join([rvd[y] for y in res])

    def build_model(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

    def export_model_to_pb(self):
        pass
