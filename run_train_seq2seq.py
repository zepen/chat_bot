# -*- coding:utf-8 -*-
"""
训练seq2seq模型
"""
import os
import time
import tensorflow as tf
from multiprocessing import cpu_count
from algorithm.seq2seq import HyperParameters, Seq2Seq
from joblib import dump

model_type = "seq2seq"
tf.flags.DEFINE_string('device', 'cpu', "设定训练设备")
tf.flags.DEFINE_integer('gpu_no', 0, "设置使用GPU编号")
tf.flags.DEFINE_float("learning_rate", 0.002, "初始学习速率")
tf.flags.DEFINE_float("clip_norm", 5, "梯度裁剪率")
tf.flags.DEFINE_integer('epoch', 20, "训练迭代轮数")
tf.flags.DEFINE_integer('embedding_size', 256, "嵌入层节点数目")
tf.flags.DEFINE_integer('encoder_hidden_units', 256, "编码层隐藏节点数目")
tf.flags.DEFINE_float('encoder_keep_prob', 0.75, "编码层保持节点")
tf.flags.DEFINE_integer('decoder_hidden_units', 256, "解码层隐藏节点数目")
tf.flags.DEFINE_float('decoder_keep_prob', 0.75, "解码保持节点")
tf.flags.DEFINE_integer('layer_num', 3, "隐藏层数目")
tf.flags.DEFINE_integer('beam_search', 0, "是否启用beam_search")
tf.flags.DEFINE_integer('beam_size', 100, "beam_search大小")
tf.flags.DEFINE_integer('batch_size', 64, "训练批次大小")
tf.flags.DEFINE_integer("decay_steps", 1000, "多少步学习速率衰减一次")
tf.flags.DEFINE_float("decay_rate", 0.95, "学习速率衰减率")
tf.flags.DEFINE_string("mode", "train", "图模式")
tf.flags.DEFINE_integer("max_decode_len", 100, "最大解码长度")
tf.flags.DEFINE_integer("save_step", 100, "间隔多少step保存一次模型")
tf.flags.DEFINE_string("model_version", "003", "模型版本")

FLAGS = tf.flags.FLAGS

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(
    device_count={"CPU": int(cpu_count())},
    log_device_placement=False,
    gpu_options=gpu_options
)

hp = HyperParameters()
hp.device = FLAGS.device
hp.gpu_no = FLAGS.gpu_no
hp.lr = FLAGS.learning_rate
hp.decay_steps = FLAGS.decay_steps
hp.decay_rate = FLAGS.decay_rate
hp.clip_norm = FLAGS.clip_norm
hp.embedding_size = FLAGS.embedding_size
hp.encoder_hidden_units = FLAGS.encoder_hidden_units
hp.encoder_keep_prob = FLAGS.encoder_keep_prob
hp.decoder_hidden_units = FLAGS.decoder_hidden_units
hp.decoder_keep_prob = FLAGS.decoder_keep_prob
hp.layer_num = FLAGS.layer_num
hp.beam_search = FLAGS.beam_search
hp.beam_size = FLAGS.beam_size
hp.mode = FLAGS.mode
hp.max_decode_len = FLAGS.max_decode_len

# 持久化超参数到文件
with open("logs/{}_hyper_parameters.pkl".format(model_type), "wb") as f:
    dump(hp, f)


def train_model():
    seq2seq_train = Seq2Seq(hp=hp)
    seq2seq_train.data_set.iterator_func(FLAGS.epoch, FLAGS.batch_size)

    def get_batch_():
        get_ = sess.run(seq2seq_train.data_set.iterator.get_next())
        encoder_inputs = get_["encoder_inputs"]
        decoder_inputs = get_["decoder_inputs"]
        decoder_target = get_["decoder_target"]
        encoder_inputs_length = [len(s[s > 0]) for s in encoder_inputs]
        decoder_targets_length = [len(s[s > 0]) for s in decoder_target]
        return encoder_inputs, decoder_inputs, decoder_target, encoder_inputs_length, decoder_targets_length

    with tf.Session(config=config) as sess:
        if not os.path.exists("logs/{}/".format(model_type)):
            os.mkdir("logs/{}/".format(model_type))
        writer = tf.summary.FileWriter("logs/{}/".format(model_type), graph=sess.graph)
        merge_all = tf.summary.merge_all()
        sess.run([tf.tables_initializer(), seq2seq_train.data_set.iterator.initializer])
        seq2seq_train.load_model(sess)
        tf.logging.info("Please open tensorboard to Supervisor train processing...")
        step = 1
        cost_time = 0
        while True:
            try:
                e_inputs, d_inputs, d_target, e_inputs_length, d_targets_length = get_batch_()
                if len(e_inputs_length) != FLAGS.batch_size:
                    break
                feed_dict = {
                    seq2seq_train.batch_size: [FLAGS.batch_size] * FLAGS.batch_size,
                    seq2seq_train.encoder_inputs_ids: e_inputs,
                    seq2seq_train.encoder_inputs_length: e_inputs_length,
                    seq2seq_train.decoder_inputs_ids: d_inputs,
                    seq2seq_train.decoder_targets_ids: d_target,
                    seq2seq_train.decoder_targets_length: d_targets_length
                }
                start_time = time.time()
                _, summary, loss = sess.run(
                    [seq2seq_train.train_op, merge_all, seq2seq_train.loss],
                    feed_dict=feed_dict
                )
                end_time = time.time()
                cost_time += (end_time - start_time)
                writer.add_summary(summary, step)
                if step % FLAGS.save_step == 0:
                    seq2seq_train.save_model(sess, save_path="logs/{}/{}.ckpt".format(model_type, model_type))
                    tf.logging.info("[step: {}, {} step/sec]".format(
                        step, round(FLAGS.save_step / cost_time, 4))
                    )
                    cost_time = 0
                step += 1
            except tf.errors.OutOfRangeError:
                tf.logging.info("All epoch is completed!")
                break


def main(_):
    if not os.path.exists("{}/".format(model_type)):
        os.mkdir("{}/".format(model_type))
    # 模型训练
    train_model()
    # 训练结束后转存为pd文件，在tensorflow-serving中加载
    tf.reset_default_graph()
    hp.mode = "predict"
    hp.encoder_keep_prob = 1.0
    hp.decoder_keep_prob = 1.0
    seq2seq_model_decode = Seq2Seq(hp=hp)
    with tf.Session() as sess:
        tf.tables_initializer().run()
        sess.run(tf.global_variables_initializer())
        seq2seq_model_decode.load_model(sess)
        seq2seq_model_decode.export_model_to_pb(
            sess,
            save_path="model/{}/".format(model_type),
            model_version=FLAGS.model_version,
        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
