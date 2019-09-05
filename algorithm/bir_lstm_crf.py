# -*- coding:utf-8 -*-
"""
定义序列标注模型
"""
import tensorflow as tf
from .base import BaseDataSet


class HyperParameters(object):
    """  超参类
    """
    device = None
    gpu_no = None
    data_set = None
    lr = 0.0
    epoch = 0
    batch_size = 0
    clip_norm = 0
    decay_steps = 0
    decay_rate = 0.0
    embedding_size = 0
    hidden_units = 0
    keep_prob = 0.0
    layer_num = 0
    num_tags = 4


class BirLstmCrfDataSet(BaseDataSet):

    def __init__(self):
        super(BirLstmCrfDataSet, self).__init__()
        self._string2index_label = tf.contrib.lookup.string_to_index_table_from_file(
            vocabulary_file=self.data_path + "label_dict.txt", num_oov_buckets=0, default_value=1)
        self._index2string_label = tf.contrib.lookup.index_to_string_table_from_file(
            vocabulary_file=self.data_path + "label_dict.txt")
        self._text_line = tf.data.TextLineDataset(self.data_path + "msr_train.txt")

    def __repr__(self):
        return "<This is BirLstmCrfDataSet>"

    def _process_func(self, string):
        """  预处理函数
        :param string: 输入句子
        :return: 词表映射后序列
        """
        line = tf.string_split([string], "\t").values
        encoder_inputs = self._string2index_table.lookup(tf.string_split([line[1]], "|").values)
        target_inputs = self._string2index_label.lookup(tf.string_split([line[2]], "|").values)
        return {
            "data": encoder_inputs,
            "label": target_inputs,
        }

    def iterator_func(self, epoch, batch_size_):
        """   获取训练批次
        :param epoch:  训练轮数
        :param batch_size_: 批次大小
        :return:  一个批次样本
        """
        data_set = self._text_line.map(self._process_func).repeat(epoch).shuffle(buffer_size=10000).\
            padded_batch(
            batch_size_,
            padded_shapes={'data': [None], "label": [None]}
        )
        self._iterator = data_set.make_initializable_iterator()


class BirLstmCrf(object):

    def __init__(self, hp):
        """  bir_lstm_crf模型用于分词
        :param hp: 模型超参数
        """
        self.num_tags = hp.num_tags
        self._set_device = "/gpu:" + str(hp.gpu_no) if hp.device == "gpu" else "/cpu:0"
        self._embedding_size = hp.embedding_size
        self._encoder_hidden_units = hp.embedding_size
        self._encoder_keep_prob = hp.encoder_keep_prob
        self._layer_num = hp.layer_num
        self._mode = hp.mode
        self._decay_steps = hp.decay_steps
        self._decay_rate = hp.decay_rate
        with tf.device(self._set_device):
            self._data_set = BirLstmCrfDataSet()
            self._build_model()
            if self._mode == "train":
                self._lr = hp.lr
                self._clip_norm = hp.clip_norm
                self._global_step = tf.Variable(0, name='global_step', trainable=False)
                self._train_func()
            elif self._mode == "predict":
                self._predict_func()
            self._saver = tf.train.Saver()

    def _inputs(self):
        self.xs = tf.placeholder(tf.int32, [None, None], name='xs')
        self.ys = tf.placeholder(tf.int32, [None, None], name='ys')
        self.sequence_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name="seq_len")
        if self._mode == "predict":
            self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.string, name='inputs_sentence')
            self.xs = self._data_set.string2index_table.lookup(
                self.encoder_inputs, name="predict_encoder_inputs"
            )

    def __repr__(self):
        return "<This is bir_lstm_crf>"

    def _encoder(self):
        # rand embedding
        embeddings = tf.get_variable(
            "embeddings",
            initializer=tf.truncated_normal_initializer(stddev=0.1),
            shape=(self._data_set.vocab_size + 1,  self._embedding_size))
        look_embed = tf.nn.embedding_lookup(embeddings, self.xs)

        fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self._encoder_hidden_units, name="fw_lstm_cell")
        bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self._encoder_hidden_units, name="bw_lstm_cell")

        fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell, output_keep_prob=self._encoder_keep_prob)
        bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell, output_keep_prob=self._encoder_keep_prob)

        cell_forward = tf.nn.rnn_cell.MultiRNNCell([fw_lstm_cell for _ in range(self._layer_num)])
        cell_backward = tf.nn.rnn_cell.MultiRNNCell([bw_lstm_cell for _ in range(self._layer_num)])

        outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            cell_forward, cell_backward, inputs=look_embed, dtype=tf.float32
        )
        forward_out, backward_out = outputs
        self._encoder_output = tf.concat([forward_out, backward_out], axis=2, name="encoder_output")

    def _decoder(self):
        weights = tf.get_variable("weights", [2 * self._embedding_size, self.num_tags])
        matricized_x_t = tf.reshape(self._encoder_output, [-1, 2 * self._embedding_size])
        matricized_unary_scores = tf.matmul(matricized_x_t, weights)
        encoder_shape = tf.shape(self._encoder_output)
        self._unary_scores = tf.reshape(
            matricized_unary_scores, [-1, encoder_shape[1], self.num_tags], name="unary_scores"
        )

        self._log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self._unary_scores, self.ys, self.sequence_lengths
        )
        tf.summary.histogram("transition_params", self.transition_params)

    def _build_model(self):
        with tf.name_scope("inputs"):
            self._inputs()
        with tf.name_scope("encoder"):
            self._encoder()
        with tf.name_scope("decoder"):
            self._decoder()

    def _train_func(self):
        with tf.name_scope("train"):
            with tf.name_scope("loss"):
                self._loss = tf.reduce_mean(-self._log_likelihood, name="loss")
            tf.summary.scalar("train_loss", self._loss)

            with tf.name_scope("train_op"):
                self._learning_rate = tf.train.exponential_decay(
                    learning_rate=self._lr,
                    global_step=self._global_step,
                    decay_steps=self._decay_steps,
                    decay_rate=self._decay_rate,
                    staircase=True
                )
                tf.summary.scalar("learning_rate", self._learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
                grads, variables = zip(*optimizer.compute_gradients(self._loss))
                grads, global_norm = tf.clip_by_global_norm(grads, self._clip_norm)
                self._train_op = optimizer.apply_gradients(zip(grads, variables), global_step=self._global_step)

    def _predict_func(self):
        with tf.name_scope("predict"):
            self._prediction, viterbi_score = tf.contrib.crf.crf_decode(
                self._unary_scores, self.transition_params,
                self.sequence_lengths
            )

    @property
    def data_set(self):
        return self._data_set

    @property
    def prediction(self):
        return self._prediction

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

    def load_model(self, sess, model_name):
        """  加载模型

        :param sess:
        :param model_name: 模型名称
        :return:
        """
        try:
            self._saver.restore(sess, tf.train.latest_checkpoint("logs/{}".format(model_name)))
        except Exception as e:
            sess.run(tf.global_variables_initializer())
            tf.logging.error("[LOAD_MODEL] {}".format(str(e)))

    @staticmethod
    def load_pb_model(sess, model_name, model_version):
        """  加载pb模型

        :param sess:
        :param model_name: 模型名称
        :param model_version: 模型版本
        :return:
        """
        try:
            tf.saved_model.loader.load(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                "model/{}/{}".format(model_name, model_version)
            )
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
            "encoder_inputs": tf.saved_model.utils.build_tensor_info(self.encoder_inputs),
            "encoder_inputs_length": tf.saved_model.utils.build_tensor_info(self.sequence_lengths)
        }
        output = {"prediction": tf.saved_model.utils.build_tensor_info(self._prediction)}
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
