# -*- coding:utf-8 -*-
"""
训练seq2seq模型
"""
import os
import time
import tensorflow as tf
from multiprocessing import cpu_count
from algorithm.bir_lstm_crf import HyperParameters, BirLstmCrf
from joblib import dump

model_type = "bir_lstm_crf"
tf.flags.DEFINE_string('device', 'cpu', "设定训练设备")
tf.flags.DEFINE_integer('gpu_no', 0, "设置使用GPU编号")
tf.flags.DEFINE_float("learning_rate", 0.001, "初始学习速率")
tf.flags.DEFINE_float("clip_norm", 5, "梯度裁剪率")
tf.flags.DEFINE_integer('epoch', 20, "训练迭代轮数")
tf.flags.DEFINE_integer('embedding_size', 256, "嵌入层节点数目")
tf.flags.DEFINE_integer('encoder_hidden_units', 256, "编码层隐藏节点数目")
tf.flags.DEFINE_float('encoder_keep_prob', 0.75, "编码层保持节点")
tf.flags.DEFINE_integer('layer_num', 3, "隐藏层数目")
tf.flags.DEFINE_integer('batch_size', 64, "训练批次大小")
tf.flags.DEFINE_integer("decay_steps", 1000, "多少步学习速率衰减一次")
tf.flags.DEFINE_float("decay_rate", 0.95, "学习速率衰减率")
tf.flags.DEFINE_string("mode", "train", "图模式")
tf.flags.DEFINE_integer("max_len", 300, "最大序列长度")
tf.flags.DEFINE_integer("save_step", 100, "间隔多少step保存一次模型")
tf.flags.DEFINE_string("model_version", "001", "模型版本")

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
hp.layer_num = FLAGS.layer_num
hp.mode = FLAGS.mode

# 持久化超参数到文件
with open("logs/{}_hyper_parameters.pkl".format(model_type), "wb") as f:
    dump(hp, f)


def train_model():
    bir_lstm_crf_train = BirLstmCrf(hp=hp)
    bir_lstm_crf_train.data_set.iterator_func(FLAGS.epoch, FLAGS.batch_size)

    def get_batch_():
        get_ = sess.run(bir_lstm_crf_train.data_set.iterator.get_next())
        encoder_inputs = get_["data"]
        target_inputs = get_["label"]
        encoder_inputs_length = [len(s[s > 0]) for s in encoder_inputs]
        return encoder_inputs, encoder_inputs_length, target_inputs

    with tf.Session(config=config) as sess:
        if not os.path.exists("logs/{}/".format(model_type)):
            os.mkdir("logs/{}/".format(model_type))
        writer = tf.summary.FileWriter("logs/{}/".format(model_type), graph=sess.graph)
        merge_all = tf.summary.merge_all()
        sess.run([tf.tables_initializer(), bir_lstm_crf_train.data_set.iterator.initializer])
        bir_lstm_crf_train.load_model(sess, model_type)
        tf.logging.info("Please open tensorboard to Supervisor train processing...")
        step = 1
        cost_time = 0
        while True:
            try:
                e_inputs, e_inputs_length, target = get_batch_()
                if len(e_inputs_length) != FLAGS.batch_size:
                    break
                feed_dict = {
                    bir_lstm_crf_train.xs: e_inputs,
                    bir_lstm_crf_train.sequence_lengths: e_inputs_length,
                    bir_lstm_crf_train.ys: target,
                }
                start_time = time.time()
                _, summary, loss = sess.run(
                    [bir_lstm_crf_train.train_op, merge_all, bir_lstm_crf_train.loss],
                    feed_dict=feed_dict
                )
                end_time = time.time()
                cost_time += (end_time - start_time)
                writer.add_summary(summary, step)
                if step % FLAGS.save_step == 0:
                    bir_lstm_crf_train.save_model(sess, save_path="logs/{}/{}.ckpt".format(model_type, model_type))
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
    seq2seq_model_decode = BirLstmCrf(hp=hp)
    with tf.Session() as sess:
        tf.tables_initializer().run()
        sess.run(tf.global_variables_initializer())
        seq2seq_model_decode.load_model(sess, model_type)
        seq2seq_model_decode.export_model_to_pb(
            sess,
            save_path="model/{}/".format(model_type),
            model_version=FLAGS.model_version,
        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
