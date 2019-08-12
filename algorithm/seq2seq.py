# -*- coding:utf-8 -*-
"""
定义seq2seq模型
"""
import tensorflow as tf


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
    embedding_size = 0
    encoder_hidden_units = 0
    encoder_keep_prob = 0.0
    decoder_hidden_units = 0
    decoder_keep_prob = 0.0
    layer_num = 0
    beam_search = 0
    beam_size = 0
    mode = ""
    max_decode_len = 0


class DataSet(object):

    def __init__(self):
        self.data_path = "./data/"
        with tf.gfile.GFile(self.data_path + "vocab_dict.txt") as f:
            self._vocab_size = len(f.readlines())
        self._string2index_table = tf.contrib.lookup.string_to_index_table_from_file(
            vocabulary_file=self.data_path + "vocab_dict.txt", num_oov_buckets=0, default_value=1)
        self._index2string_table = tf.contrib.lookup.index_to_string_table_from_file(
            vocabulary_file=self.data_path + "vocab_dict.txt")
        self._text_line = tf.data.TextLineDataset(self.data_path + "train_corpus.txt")
        self._iterator = None

    def _process_func(self, string):
        """  预处理函数
        :param string: 输入句子
        :return: 词表映射后序列
        """
        line = tf.string_split([string], "\t").values
        encoder_inputs = self._string2index_table.lookup(tf.string_split([line[1]], "|").values)
        decoder_inputs = self._string2index_table .lookup(tf.string_split(["_GO_|" + line[2]], "|").values)
        decoder_target = self._string2index_table .lookup(tf.string_split([line[2] + "|_EOS_"], "|").values)
        return {
            "encoder_inputs": encoder_inputs,
            "decoder_inputs": decoder_inputs,
            "decoder_target": decoder_target
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
            padded_shapes={'encoder_inputs': [None], 'decoder_inputs': [None], "decoder_target": [None]}
        )
        self._iterator = data_set.make_initializable_iterator()

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def string2index_table(self):
        return self._string2index_table

    @property
    def index2string_table(self):
        return self._index2string_table

    @property
    def iterator(self):
        return self._iterator


class Seq2Seq(object):

    def __init__(self, hp):
        """  seq2seq模型用于生成回复
        :param hp: 模型超参数
        """
        self._set_device = "/gpu:" + str(hp.gpu_no) if hp.device == "gpu" else "/cpu:0"
        self._embedding_size = hp.embedding_size
        self._encoder_hidden_units = hp.embedding_size
        self._encoder_keep_prob = hp.encoder_keep_prob
        self._decoder_hidden_units = hp.decoder_hidden_units
        self._decoder_keep_prob = hp.decoder_keep_prob
        self._layer_num = hp.layer_num
        self._beam_search = hp.beam_search
        self._beam_size = hp.beam_size
        self._mode = hp.mode
        self._max_decode_len = hp.max_decode_len
        with tf.device(self._set_device):
            self._data_set = DataSet()
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
        # encoder_inputs
        self.encoder_inputs_ids = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')
        # decoder_inputs
        self.decoder_inputs_ids = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
        self.decoder_targets_ids = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
        # batch_size
        self.batch_size = tf.placeholder(shape=[None], dtype=tf.int32, name='batch_size')
        self._max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        self._mask = tf.sequence_mask(
            self.decoder_targets_length, self._max_target_sequence_length, dtype=tf.float32, name='masks')
        self._max_len_index = tf.argmax(self.decoder_targets_length, name="max_len_index")
        if self._mode == "predict":
            self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.string, name='inputs_sentence')
            self.encoder_inputs_ids = self._data_set.string2index_table.lookup(
                self.encoder_inputs, name="predict_encoder_inputs"
            )

    def _encoder(self):
        with tf.name_scope("embedding"):
            self._embeddings = tf.Variable(
                tf.concat((
                    tf.zeros((1, self._embedding_size), dtype=tf.float32),
                    tf.random_uniform([self._data_set.vocab_size + 1, self._embedding_size], -1.0, 1.0)), axis=0),
                dtype=tf.float32)
            self._encoder_inputs_embedded = tf.nn.embedding_lookup(self._embeddings, self.encoder_inputs_ids)
            self._decoder_inputs_embedded = tf.nn.embedding_lookup(self._embeddings, self.decoder_inputs_ids)

        with tf.name_scope("encoder_lstm"):
            encoder_cell = tf.nn.rnn_cell.LSTMCell(self._encoder_hidden_units, name="encoder_cell")
            encoder_cell = tf.nn.rnn_cell.DropoutWrapper(encoder_cell, output_keep_prob=self._encoder_keep_prob)
            cell_forward = tf.nn.rnn_cell.MultiRNNCell([encoder_cell for _ in range(self._layer_num)])
            cell_backward = tf.nn.rnn_cell.MultiRNNCell([encoder_cell for _ in range(self._layer_num)])
            bi_output, bi_state = \
                tf.nn.bidirectional_dynamic_rnn(
                    cell_forward, cell_backward, inputs=self._encoder_inputs_embedded, dtype=tf.float32
                )
            self._encoder_outputs = tf.concat([bi_output[0], bi_output[1]], axis=2, name="encoder_output")
            encoder_state_c, encoder_state_h = (
                tf.concat(
                    [
                        tf.reduce_mean([bi_state[0][i][j] for i in range(self._layer_num)], axis=0),
                        tf.reduce_mean([bi_state[1][i][j] for i in range(self._layer_num)], axis=0)
                    ], axis=1, name="state_" + e) for j, e in enumerate(["c", "h"]))
            self._encoder_state = tf.nn.rnn_cell.LSTMStateTuple(encoder_state_c, encoder_state_h)

    def _decoder(self):
        if self._mode == "predict" and self._beam_search:
            tf.logging.info("use beam_search decoding.")
            self._encoder_outputs = tf.contrib.seq2seq.tile_batch(
                self._encoder_outputs, multiplier=self._beam_size)
            self._encoder_state = tf.contrib.seq2seq.tile_batch(
                self._encoder_state, multiplier=self._beam_size)
            self._encoder_inputs_length = tf.contrib.seq2seq.tile_batch(
                self._encoder_inputs_length, multiplier=self._beam_size)

        with tf.name_scope("attention"):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self._encoder_hidden_units,
                memory=self._encoder_outputs,
                memory_sequence_length=self.encoder_inputs_length
            )

        with tf.name_scope("decoder_lstm"):
            decoder_cell = tf.nn.rnn_cell.LSTMCell(2 * self._decoder_hidden_units, name="decoder_cell")
            decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob=self._decoder_keep_prob)
            self._decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=decoder_cell,
                attention_mechanism=attention_mechanism,
                attention_layer_size=self._decoder_hidden_units,
                name='attention_wrapper'
            )

        if self._beam_search:
            self._decoder_initial_state = self._decoder_cell.zero_state(
                batch_size=self.batch_size[0] * self._beam_size, dtype=tf.float32).clone(
                cell_state=self._encoder_state)
        else:
            self._decoder_initial_state = self._decoder_cell.zero_state(
                batch_size=self.batch_size[0], dtype=tf.float32).clone(
                cell_state=self._encoder_state
            )

        with tf.name_scope("output_layer"):
            self._output_layer = tf.layers.Dense(
                self._data_set.vocab_size + 2,
                kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
            )

        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=self._decoder_inputs_embedded,
            sequence_length=self.decoder_targets_length,
            time_major=False,
            name='training_helper'
        )
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self._decoder_cell,
            helper=training_helper,
            initial_state=self._decoder_initial_state,
            output_layer=self._output_layer
        )
        self._decoder_outputs, _, _ = \
            tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder,
                impute_finished=True,
                maximum_iterations=self._max_target_sequence_length
            )

    def _build_model(self):
        with tf.name_scope("inputs"):
            self._inputs()
        with tf.name_scope("encoder"):
            self._encoder()
        with tf.name_scope("decoder"):
            self._decoder()

    def _train_func(self):
        with tf.name_scope("train"):
            with tf.name_scope("logits"):
                self._logits = tf.identity(self._decoder_outputs.rnn_output)

            with tf.name_scope("loss"):
                self._loss = tf.contrib.seq2seq.sequence_loss(
                    logits=self._logits, targets=self.decoder_targets_ids, weights=self._mask)
            tf.summary.scalar("train_loss", self._loss)

            with tf.name_scope("train_op"):
                optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
                grads, variables = zip(*optimizer.compute_gradients(self._loss))
                grads, global_norm = tf.clip_by_global_norm(grads, self._clip_norm)
                self._train_op = optimizer.apply_gradients(zip(grads, variables), global_step=self._global_step)

    def _predict_func(self):
        with tf.name_scope("predict"):
            start_tokens = tf.cast(
                tf.ones(
                    [self.batch_size[0]], tf.int64) * self._data_set.string2index_table.lookup(tf.constant(["_GO_"])),
                dtype=tf.int32
            )
            end_token = tf.cast(self._data_set.string2index_table.lookup(tf.constant(["_EOS_"])), dtype=tf.int32)[0]
            if self._beam_search:
                inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=self._decoder_cell,
                    embedding=self._embeddings,
                    start_tokens=start_tokens,
                    end_token=end_token,
                    initial_state=self._decoder_initial_state,
                    beam_width=self._beam_size,
                    output_layer=self._output_layer
                )
            else:
                decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=self._embeddings, start_tokens=start_tokens, end_token=end_token
                )
                inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=self._decoder_cell, helper=decoding_helper,
                    initial_state=self._decoder_initial_state,
                    output_layer=self._output_layer
                )
            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=inference_decoder, maximum_iterations=self._max_decode_len
            )
            if self._beam_search:
                with tf.name_scope("prediction"):
                    self._prediction = self._data_set.index2string_table.lookup(
                        tf.cast(
                            tf.transpose(decoder_outputs.predicted_ids, perm=[0, 2, 1]),
                            dtype=tf.int64
                        )
                    )
            else:
                with tf.name_scope("prediction"):
                    self._prediction = self._data_set.index2string_table.lookup(
                        tf.cast(decoder_outputs.sample_id, dtype=tf.int64)
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

    def load_model(self, sess):
        """  加载模型

        :param sess:
        :return:
        """
        try:
            self._saver.restore(sess, tf.train.latest_checkpoint("logs/model/"))
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
            "encoder_inputs": tf.saved_model.utils.build_tensor_info(self.encoder_inputs),
            "encoder_inputs_length": tf.saved_model.utils.build_tensor_info(self.encoder_inputs_length),
            "batch_size": tf.saved_model.utils.build_tensor_info(self.batch_size)
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
