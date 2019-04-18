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

        with tf.device(set_device):
            with tf.name_scope("inputs"):
                self._encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
                self._decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
                self._decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

        with tf.device('/cpu:0'):
            with tf.name_scope("embedding"):
                embeddings = tf.Variable(
                    tf.random_uniform([kwargs["vocab_size"], hp.input_embedding_size], -1.0, 1.0), dtype=tf.float32)
                self._encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self._encoder_inputs)
                self._decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self._decoder_inputs)

        with tf.device(set_device):
            with tf.name_scope("encoder"):
                encoder_cell = tf.contrib.rnn.LSTMCell(hp.encoder_hidden_units)
                with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                    self._encoder_outputs, self._encoder_final_state = \
                        tf.nn.dynamic_rnn(
                            encoder_cell,
                            self._encoder_inputs_embedded,
                            dtype=tf.float32,
                        )
            with tf.name_scope("decoder"):
                decoder_cell = tf.contrib.rnn.LSTMCell(hp.decoder_hidden_units)
                with tf.variable_scope("", reuse=tf.AUTO_REUSE):
                    self._decoder_outputs, self._decoder_final_state = \
                        tf.nn.dynamic_rnn(
                            decoder_cell,
                            self._decoder_inputs_embedded,
                            initial_state=self._encoder_final_state,
                            dtype=tf.float32,
                        )
            with tf.name_scope("full_connect"):
                decoder_logits = tf.contrib.layers.linear(self._decoder_outputs, hp.vocab_size)

            with tf.name_scope("softmax"):
                self._cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.one_hot(self._decoder_targets, depth=hp.vocab_size, dtype=tf.float32),
                    logits=decoder_logits,
                )
            with tf.name_scope("loss"):
                self._loss = tf.reduce_mean(self._cross_entropy)
                tf.summary.scalar("train_loss", self._loss)
            with tf.name_scope("trian_op"):
                self._train_op = tf.train.AdamOptimizer().minimize(self._loss)

            with tf.device('/cpu:0'):
                with tf.name_scope("predict"):
                    self._decoder_prediction = tf.argmax(decoder_logits, 2)

    @property
    def encoder_inputs(self):
        return self._encoder_inputs

    @property
    def decoder_inputs(self):
        return self._decoder_inputs

    @property
    def decoder_targets(self):
        return self._decoder_targets

    @property
    def decoder_prediction(self):
        return self._decoder_prediction

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
        decode_len = 0
        while 1:
            con_tf_s.calculate_predict_result(data)
            predict_res = con_tf_s.predict_result["predictions"][0][-1]
            if rvd[predict_res] == "_EOS_":
                break
            if decode_len > m_config.max_decode_len:
                return m_config.replace_sentence
            res.append(predict_res)
            decode_len += 1
            data["instances"][0]["decoder_inputs"].append(predict_res)
        output_text = rule_correction("".join([rvd[y] for y in res]))
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
            "decoder_inputs": tf.saved_model.utils.build_tensor_info(kwargs["decoder_inputs"]),
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
