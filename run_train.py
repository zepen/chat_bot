# -*- coding:utf-8 -*-
"""
训练seq2seq模
"""
import tensorflow as tf
from utils.model_config import ModelConfig

tf.app.flags.DEFINE_string('device', 'cpu', "设定训练设备")
tf.app.flags.DEFINE_integer('train_steps_num', 1000, "设置迭代次数")
tf.app.flags.DEFINE_boolean('bool_name', False, "descript3")
FLAGS = tf.app.flags.FLAGS

mc = ModelConfig()
mc.device = FLAGS.device
mc.train_steps_num = FLAGS.train_steps_num

gpu_options = tf.GPUOptions(allow_growth=mc.gpu_options)
config = tf.ConfigProto(
    device_count={"CPU": 8},
    log_device_placement=True,
    gpu_options=gpu_options
)


def main(_):
    with tf.Session(config=config) as sess:
        pass


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
