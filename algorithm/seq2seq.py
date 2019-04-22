# -*- coding:utf-8 -*-
"""
定义seq2seq模型
"""
import tensorflow as tf


class Seq2SeqModel(object):

    def __init__(self, hp, **kwargs):
        """ seq2seq 模型用于生成对话

        :param hp: 模型超参数
        """
        set_device = None
        if hp.device == "cpu":
            set_device = "/cpu:0"
        elif hp.device == "gpu":
            set_device = "/gpu:" + hp.gpu_no

        self._vocab_dict = kwargs["vocab_dict"]  # 用于解码
        self._r_vocab_dict = kwargs["r_vocab_dict"]  # 用于映射解码结果

        with tf.device(set_device):
            with tf.name_scope("inputs"):
                self._encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
                self._encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')
                self._decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
                self._decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
                self._decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
                self._max_target_sequence_length = tf.reduce_max(self._decoder_targets_length, name='max_target_len')
                self._mask = tf.sequence_mask(
                    self._decoder_targets_length, self._max_target_sequence_length, dtype=tf.float32, name='masks')
                self._batch_size = tf.placeholder(shape=[None], dtype=tf.int32, name='batch_size')

        with tf.device('/cpu:0'):
            with tf.name_scope("embedding"):
                embeddings = tf.Variable(
                    tf.random_uniform([kwargs["vocab_size"], hp.input_embedding_size], -1.0, 1.0), dtype=tf.float32)
                self._encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self._encoder_inputs)
                self._decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self._decoder_inputs)

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
                    self._encoder_output = tf.concat([forward_out, backward_out], axis=2, name="encoder_output")
                    encoder_state_c = tf.concat(
                        [forward_state[-1][0], backward_state[-1][0]], axis=1, name="state_c")
                    encoder_state_h = tf.concat(
                        [forward_state[-1][1], backward_state[-1][1]], axis=1, name="state_h")
                    self._encoder_final_state = tf.nn.rnn_cell.LSTMStateTuple(encoder_state_c, encoder_state_h)

            with tf.name_scope("attention"):
                with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                        num_units=hp.encoder_hidden_units,
                        memory=self._encoder_output,
                        memory_sequence_length=self._encoder_inputs_length
                    )

            with tf.name_scope("decoder"):
                with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                    decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=2*hp.decoder_hidden_units)
                    decoder_cell = tf.nn.rnn_cell.DropoutWrapper(
                        decoder_cell, output_keep_prob=hp.decoder_keep_prob)
                    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                        cell=decoder_cell,
                        attention_mechanism=attention_mechanism,
                        attention_layer_size=hp.decoder_hidden_units,
                        name='attention_wrapper')

                decoder_initial_state = decoder_cell.zero_state(
                    batch_size=self._batch_size[0], dtype=tf.float32).clone(cell_state=self._encoder_final_state)

                with tf.name_scope("output_layer"):
                    output_layer = tf.layers.Dense(
                        hp.vocab_size,
                        kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
                    )

                training_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=self._decoder_inputs_embedded,
                    sequence_length=self._decoder_targets_length,
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
                        impute_finished=False,
                        maximum_iterations=self._max_target_sequence_length)

            with tf.name_scope("train"):
                with tf.name_scope("logits"):
                    self._logits = tf.identity(self._decoder_outputs.rnn_output)

                with tf.name_scope("loss"):
                    self._loss = tf.contrib.seq2seq.sequence_loss(
                        logits=self._logits, targets=self._decoder_targets, weights=self._mask)
                tf.summary.scalar("train_loss", self._loss)

                with tf.name_scope("trian_op"):
                    self._train_op = tf.train.AdamOptimizer().minimize(self._loss)

            with tf.name_scope("predict"):
                start_tokens = tf.ones([self._batch_size[0]], tf.int32) * self._vocab_dict['_GO_']
                end_token = self._vocab_dict['_EOS_']
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
                with tf.name_scope("prediction"):
                    self._decoder_prediction = decoder_outputs.sample_id

            if kwargs["show_text"]:
                tf.summary.text('encoder_inputs',
                                tf.py_func(self._index_to_text, [self._encoder_inputs[0]], tf.string))
                tf.summary.text('decoder_inputs',
                                tf.py_func(self._index_to_text, [self._decoder_inputs[0]], tf.string))
                tf.summary.text('targets_predict',
                                tf.py_func(self._index_to_text, [self._decoder_prediction[0]], tf.string))

    def _index_to_text(self, input_tensor):
        """

        :param input_tensor: 输入张量
        :return: numpy.array
        """
        if len(self._r_vocab_dict):
            return "".join([self._r_vocab_dict[index]
                            for index in input_tensor if self._r_vocab_dict.get(index) and index != 0])
        else:
            raise Exception("No load r_vocab_dict!")

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

    @staticmethod
    def predict_func(input_text, vd, rvd, con_tf_s, m_config, rule_correction):
        """  解码预测

        :param input_text 输出文本
        :param vd: 词表
        :param rvd: 反转词表
        :param con_tf_s 连接 tf_serving object
        :param m_config 配置对象
        :param rule_correction 规则修正对象
        :return: str
        """
        x = [vd[x] if vd.get(x) else vd["_UNK_"] for x in list(input_text)]
        data = {
            "instances": [
                {
                    "encoder_inputs": x,
                    "encoder_inputs_length": len(x),
                    "batch_size": 1
                },
            ]
        }
        con_tf_s.calculate_predict_result(data)
        predict_res = con_tf_s.predict_result["predictions"][0]
        output_text = rule_correction("".join([rvd[y] for y in predict_res if rvd[y] != "_EOS_"]))
        return m_config.replace_sentence if output_text is None else output_text

    @staticmethod
    def save_model(saver, sess, save_path, gs):
        """  保存模型

        :param saver: 存放对象
        :param sess:  会话
        :param save_path:  模型存放路径
        :param gs:  global step
        :return:
        """
        try:
            saver.save(sess, save_path, gs)
        except Exception as e:
            tf.logging.error("[SAVER_MODEL] {}".format(str(e)))

    @staticmethod
    def load_model(sess, meta_path, checkpoint_path):
        """  加载模型

        :param sess:
        :param meta_path:
        :param checkpoint_path:
        :return:
        """
        saver = tf.train.import_meta_graph(meta_path)
        try:
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        except Exception as e:
            tf.logging.error("[LOAD_MODEL] {}".format(str(e)))

    @staticmethod
    def export_model_to_pb(sess, save_path, model_version, **kwargs):
        """ 保存模型为pb格式

        :param sess: 会话
        :param save_path: 模型存放路径
        :param model_version: 模型版本
        :param kwargs:
        :return:
        """
        builder = tf.saved_model.builder.SavedModelBuilder(save_path + model_version)
        inputs = {
            "encoder_inputs": tf.saved_model.utils.build_tensor_info(kwargs["encoder_inputs"]),
            "encoder_inputs_length": tf.saved_model.utils.build_tensor_info(kwargs["encoder_inputs_length"]),
            "batch_size": tf.saved_model.utils.build_tensor_info(kwargs["batch_size"])
        }
        output = {"predictions": tf.saved_model.utils.build_tensor_info(kwargs["predictions"])}
        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=output,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature}
        )
        try:
            builder.save()
        except Exception as e:
            tf.logging.error("[PB_SAVE_MODEL] {}".format(str(e)))
