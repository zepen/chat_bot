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
tf.app.flags.DEFINE_integer('train_steps_num', 1000, "设置迭代次数")
tf.app.flags.DEFINE_integer('batch_size', 32, "训练批次大小")
tf.app.flags.DEFINE_string("model_version", "001", "模型版本")
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
seq2seq_model = Seq2SeqModel(
    mc,
    batch_size=mc.batch_size,
    vocab_size=mc.vocab_size,
    mode="train",
    vocab_dict=processing_corpus.vocab_dict,
    r_vocab_dict=processing_corpus.r_vocab_dict,
    show_text=False
)


def main(_):
    if not os.path.exists("model/"):
        os.mkdir("model/")
    if os.path.exists("model/" + mc.model_version):
        os.remove("model/" + mc.model_version + "/")
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter("logs/", graph=sess.graph)
        merge_all = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        tf.logging.info("Please open tensorboard to Supervisor train processing...")
        for step in range(FLAGS.train_steps_num):
            encoder_inputs, decoder_inputs, decoder_target, encoder_inputs_length, decoder_targets_length \
                = processing_corpus.get_batch(mc.batch_size)
            _, summary, loss = sess.run(
                [seq2seq_model.train_op, merge_all, seq2seq_model.loss],
                feed_dict={
                    seq2seq_model.batch_size: [mc.batch_size] * mc.batch_size,
                    seq2seq_model.encoder_inputs: encoder_inputs,
                    seq2seq_model.encoder_inputs_length: encoder_inputs_length,
                    seq2seq_model.decoder_inputs: decoder_inputs,
                    seq2seq_model.decoder_targets: decoder_target,
                    seq2seq_model.decoder_targets_length: decoder_targets_length
                }
            )
            writer.add_summary(summary, step)
            if step % 1000 == 0:
                seq2seq_model.save_model(saver, sess, save_path="logs/", gs=step)
        seq2seq_model.export_model_to_pb(
            sess,
            save_path="model/",
            model_version=mc.model_version,
            encoder_inputs=seq2seq_model.encoder_inputs,
            encoder_inputs_length=seq2seq_model.encoder_inputs_length,
            batch_size=seq2seq_model.batch_size,
            predictions=seq2seq_model.decoder_prediction
        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
