# -*- coding:utf-8 -*-
"""
定义seq2seq模型
"""
import tensorflow as tf


class Seq2SeqModel(object):

    def __init__(self, hp, **kwargs):
        """ seq2seq 模型用于生成回复

        :param hp: 模型超参数
        """
        set_device = None
        if hp.device == "cpu":
            set_device = "/cpu:0"
        elif hp.device == "gpu":
            set_device = "/gpu:" + hp.gpu_no
        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        with tf.device(set_device):
            with tf.name_scope("inputs"):
                self._encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.string, name='encoder_inputs')
                self._encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')
                self._decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.string, name='decoder_inputs')
                self._decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.string, name='decoder_targets')
                self._decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
                self._max_target_sequence_length = tf.reduce_max(self._decoder_targets_length, name='max_target_len')
                self._mask = tf.sequence_mask(
                    self._decoder_targets_length, self._max_target_sequence_length, dtype=tf.float32, name='masks')
                self._batch_size = tf.placeholder(shape=[None], dtype=tf.int32, name='batch_size')
                self._max_len_index = tf.argmax(self._decoder_targets_length, name="max_len_index")

        with tf.name_scope("map_sequence"):
            self._string2index_table = tf.contrib.lookup.string_to_index_table_from_file(
                vocabulary_file="./dictionary/vocab_dict.txt", num_oov_buckets=0, default_value=1)
            self._index2string_table = tf.contrib.lookup.index_to_string_table_from_file(
                vocabulary_file="./dictionary/vocab_dict.txt")
            self._encoder_inputs_ids = self._string2index_table.lookup(self._encoder_inputs)
            self._decoder_inputs_ids = self._string2index_table.lookup(self._decoder_inputs)
            self._decoder_targets_ids = self._string2index_table.lookup(self._decoder_targets)

        with tf.device('/cpu:0'):
            with tf.name_scope("embedding"):
                embeddings = tf.Variable(
                    tf.concat((
                        tf.zeros((1, hp.input_embedding_size), dtype=tf.float32),
                        tf.random_uniform([kwargs["vocab_size"] + 1, hp.input_embedding_size], -1.0, 1.0)), axis=0),
                    dtype=tf.float32)
                self._encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self._encoder_inputs_ids)
                self._decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self._decoder_inputs_ids)

        with tf.device(set_device):
            with tf.name_scope("encoder"):
                with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                    encoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=hp.encoder_hidden_units)
                    cell_forward = tf.nn.rnn_cell.MultiRNNCell([encoder_cell for _ in range(hp.layer_num)])
                    cell_backward = tf.nn.rnn_cell.MultiRNNCell([encoder_cell for _ in range(hp.layer_num)])
                    bi_output, bi_state = \
                        tf.nn.bidirectional_dynamic_rnn(
                            cell_forward, cell_backward, inputs=self._encoder_inputs_embedded, dtype=tf.float32
                        )
                    forward_state, backward_state = bi_state
                    forward_out, backward_out = bi_output
                    self._encoder_outputs = tf.concat([forward_out, backward_out], axis=2, name="encoder_output")
                    encoder_state_c = tf.concat(
                        [forward_state[-1][0], backward_state[-1][0]], axis=1, name="state_c")
                    encoder_state_h = tf.concat(
                        [forward_state[-1][1], backward_state[-1][1]], axis=1, name="state_h")
                    self._encoder_final_state = tf.nn.rnn_cell.LSTMStateTuple(encoder_state_c, encoder_state_h)

            with tf.name_scope("decoder"):
                batch_size = self._batch_size[0]
                self._beam_search = kwargs["beam_search"]
                if kwargs["mode"] == "decode" and kwargs["decode_mode"] == "beam_search":
                    if self._beam_search:
                        self._beam_size = kwargs["beam_size"]
                        tf.logging.info("use beam_search decoding.")
                        self._encoder_outputs = tf.contrib.seq2seq.tile_batch(
                            self._encoder_outputs, multiplier=self._beam_size)
                        self._encoder_final_state = tf.contrib.seq2seq.tile_batch(
                            self._encoder_final_state, multiplier=self._beam_size)
                        self._encoder_inputs_length = tf.contrib.seq2seq.tile_batch(
                            self._encoder_inputs_length, multiplier=self._beam_size)

                with tf.name_scope("attention"):
                    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                            num_units=hp.encoder_hidden_units,
                            memory=self._encoder_outputs,
                            memory_sequence_length=self._encoder_inputs_length
                        )

                with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                    if kwargs["mode"] == "train":
                        self._decoder_keep_prob = hp.decoder_keep_prob
                    elif kwargs["mode"] == "decode":
                        self._decoder_keep_prob = 1.00
                    decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=2 * hp.decoder_hidden_units)
                    decoder_cell = tf.nn.rnn_cell.DropoutWrapper(
                        decoder_cell, output_keep_prob=self._decoder_keep_prob)
                    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                        cell=decoder_cell,
                        attention_mechanism=attention_mechanism,
                        attention_layer_size=hp.decoder_hidden_units,
                        name='attention_wrapper')

                if kwargs["mode"] == "decode" and kwargs["decode_mode"] == "beam_search":
                    decoder_initial_state = decoder_cell.zero_state(
                        batch_size=batch_size * self._beam_size, dtype=tf.float32).clone(
                        cell_state=self._encoder_final_state)
                else:
                    decoder_initial_state = decoder_cell.zero_state(
                        batch_size=batch_size, dtype=tf.float32).clone(
                        cell_state=self._encoder_final_state
                    )

                with tf.name_scope("output_layer"):
                    output_layer = tf.layers.Dense(
                        kwargs["vocab_size"] + 2,
                        kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
                    )

                training_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=self._decoder_inputs_embedded,
                    sequence_length=self._decoder_targets_length,
                    time_major=False,
                    name='training_helper'
                )
                training_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell,
                    helper=training_helper,
                    initial_state=decoder_initial_state,
                    output_layer=output_layer
                )
                self._decoder_outputs, _, _ = \
                    tf.contrib.seq2seq.dynamic_decode(
                        decoder=training_decoder,
                        impute_finished=True,
                        maximum_iterations=self._max_target_sequence_length)

            if kwargs["mode"] == "train":
                with tf.name_scope("train"):
                    with tf.name_scope("logits"):
                        self._logits = tf.identity(self._decoder_outputs.rnn_output)

                    with tf.name_scope("loss"):
                        self._loss = tf.contrib.seq2seq.sequence_loss(
                            logits=self._logits, targets=self._decoder_targets_ids, weights=self._mask)
                    tf.summary.scalar("train_loss", self._loss)

                    with tf.name_scope("train_op"):
                        optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)
                        grads, variables = zip(*optimizer.compute_gradients(self._loss))
                        grads, global_norm = tf.clip_by_global_norm(grads, hp.clip_norm)
                        self._train_op = optimizer.apply_gradients(zip(grads, variables), global_step=self._global_step)

            with tf.name_scope("predict"):
                start_tokens = tf.cast(
                    tf.ones([batch_size], tf.int64) * self._string2index_table.lookup(tf.constant(["_GO_"])),
                    dtype=tf.int32
                )
                end_token = tf.cast(self._string2index_table.lookup(tf.constant(["_EOS_"])), dtype=tf.int32)[0]
                if self._beam_search and kwargs["mode"] == "decode" and kwargs["decode_mode"] == "beam_search":
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=decoder_cell,
                        embedding=embeddings,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=decoder_initial_state,
                        beam_width=self._beam_size,
                        output_layer=output_layer
                    )
                else:
                    decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        embedding=embeddings, start_tokens=start_tokens, end_token=end_token
                    )
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=decoder_cell, helper=decoding_helper,
                        initial_state=decoder_initial_state,
                        output_layer=output_layer
                    )
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                        decoder=inference_decoder, maximum_iterations=hp.max_decode_len
                )
                if self._beam_search and kwargs["mode"] == "decode" and kwargs["decode_mode"] == "beam_search":
                    with tf.name_scope("prediction"):
                        self._decoder_prediction = self._index2string_table.lookup(
                            tf.cast(
                                tf.transpose(decoder_outputs.predicted_ids, perm=[0, 2, 1]),
                                dtype=tf.int64
                            )
                        )
                else:
                    with tf.name_scope("prediction"):
                        self._decoder_prediction = self._index2string_table.lookup(
                            tf.cast(decoder_outputs.sample_id, dtype=tf.int64)
                        )

            if kwargs["mode"] == "train":
                tf.summary.text('encoder_inputs', tf.reduce_join(self._encoder_inputs[self._max_len_index]))
                tf.summary.text('decoder_inputs', tf.reduce_join(self._decoder_inputs[self._max_len_index]))
                tf.summary.text('targets_predict', tf.reduce_join(self._decoder_prediction[self._max_len_index]))

            self._saver = tf.train.Saver()

    @property
    def encoder_inputs(self):
        return self._encoder_inputs

    @property
    def encoder_inputs_length(self):
        return self._encoder_inputs_length

    @property
    def decoder_inputs(self):
        return self._decoder_inputs

    @property
    def decoder_targets_length(self):
        return self._decoder_targets_length

    @property
    def decoder_targets(self):
        return self._decoder_targets

    @property
    def decoder_prediction(self):
        return self._decoder_prediction

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def global_step(self):
        return self._global_step

    def save_model(self, sess, save_path):
        """  保存模型

        :param sess:  会话
        :param save_path:  模型存放路径
        :return:
        """
        try:
            self._saver.save(sess, save_path)
        except Exception as e:
            tf.logging.error("[SAVER_MODEL] {}".format(str(e)))

    def load_model(self, sess):
        """  加载模型

        :param sess:
        :return:
        """
        try:
            self._saver.restore(sess, tf.train.latest_checkpoint("logs/"))
        except Exception as e:
            sess.run(tf.global_variables_initializer())
            tf.logging.error("[LOAD_MODEL] {}".format(str(e)))

    @staticmethod
    def load_pb_model(sess, model_version):
        """  加载pb模型

        :param sess:
        :param model_version:
        :return:
        """
        try:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "model/" + model_version)
        except Exception as e:
            tf.logging.error("[LOAD_PB_MODEL] {}".format(str(e)))

    def export_model_to_pb(self, sess, save_path, model_version):
        """ 保存模型为pb格式

        :param sess: 会话
        :param save_path: 模型存放路径
        :param model_version: 模型版本
        :return:
        """
        builder = tf.saved_model.builder.SavedModelBuilder(save_path + model_version)
        inputs = {
            "encoder_inputs": tf.saved_model.utils.build_tensor_info(self._encoder_inputs),
            "encoder_inputs_length": tf.saved_model.utils.build_tensor_info(self._encoder_inputs_length),
            "batch_size": tf.saved_model.utils.build_tensor_info(self._batch_size)
        }
        output = {"prediction": tf.saved_model.utils.build_tensor_info(self._decoder_prediction)}
        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=output,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature},
            main_op=tf.tables_initializer()
        )
        try:
            builder.save()
        except Exception as e:
            tf.logging.error("[PB_SAVE_MODEL] {}".format(str(e)))
