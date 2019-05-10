# -*- coding:utf-8 -*-
"""
训练seq2seq模
"""
import os
import tensorflow as tf
from config.config import ModelConfig
from algorithm.processing import ProcessingCorps
from algorithm.seq2seq import Seq2SeqModel

tf.app.flags.DEFINE_string('device', 'cpu', "设定训练设备")
tf.app.flags.DEFINE_integer('cpu_num', 8, "cpu数目")
tf.app.flags.DEFINE_integer('train_steps_num', 100, "设置迭代次数")
tf.app.flags.DEFINE_integer('batch_size', 32, "训练批次大小")
tf.app.flags.DEFINE_string("model_version", "002", "模型版本")
FLAGS = tf.app.flags.FLAGS

mc = ModelConfig()
mc.device = FLAGS.device
mc.train_steps_num = FLAGS.train_steps_num
mc.batch_size = FLAGS.batch_size
mc.model_version = FLAGS.model_version


gpu_options = tf.GPUOptions(allow_growth=mc.gpu_options)
config = tf.ConfigProto(
    device_count={"CPU": int(FLAGS.cpu_num)},
    log_device_placement=True,
    gpu_options=gpu_options
)

processing_corpus = ProcessingCorps()
encoder_inputs, decoder_inputs, decoder_target, encoder_inputs_length, decoder_targets_length \
    = processing_corpus.get_batch(mc.batch_size)

def train_model():
    seq2seq_model_train = Seq2SeqModel(
        mc,
        batch_size=mc.batch_size,
        beam_search=mc.beam_search,
        beam_size=mc.beam_size,
        mode="train"
    )
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter("logs/", graph=sess.graph)
        merge_all = tf.summary.merge_all()
        try:
            saver.restore(sess, tf.train.latest_checkpoint("logs/"))
        except Exception as e:
            print(e)
            sess.run(tf.global_variables_initializer())
        tf.logging.info("Please open tensorboard to Supervisor train processing...")
        for step in range(FLAGS.train_steps_num):
            encoder_inputs, decoder_inputs, decoder_target, encoder_inputs_length, decoder_targets_length \
                = processing_corpus.get_batch(mc.batch_size)
            _, gs, summary, loss = sess.run(
                [seq2seq_model_train.train_op, seq2seq_model_train.global_step, merge_all, seq2seq_model_train.loss],
                feed_dict={
                    seq2seq_model_train.batch_size: [mc.batch_size] * mc.batch_size,
                    seq2seq_model_train.encoder_inputs: encoder_inputs,
                    seq2seq_model_train.encoder_inputs_length: encoder_inputs_length,
                    seq2seq_model_train.decoder_inputs: decoder_inputs,
                    seq2seq_model_train.decoder_targets: decoder_target,
                    seq2seq_model_train.decoder_targets_length: decoder_targets_length
                }
            )
            writer.add_summary(summary, step)
            if step % 1000 == 0:
                seq2seq_model_train.save_model(saver, sess, save_path="logs/s2s.ckpt", gs=gs // 1000)


def main(_):
    if not os.path.exists("model/"):
        os.mkdir("model/")
    train_model()
    tf.reset_default_graph()
    seq2seq_model_decode = Seq2SeqModel(
        mc,
        batch_size=mc.batch_size,
        beam_search=mc.beam_search,
        beam_size=mc.beam_size,
        decode_mode="beam_search",
        mode="decode",
    )
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        seq2seq_model_decode.load_model(sess)
        seq2seq_model_decode.export_model_to_pb(
            sess,
            save_path="model/",
            model_version=mc.model_version,
            encoder_inputs=seq2seq_model_decode.encoder_inputs,
            encoder_inputs_length=seq2seq_model_decode.encoder_inputs_length,
            batch_size=seq2seq_model_decode.batch_size,
            predictions=seq2seq_model_decode.decoder_prediction
        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
