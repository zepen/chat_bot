"""
定义模型配置
"""


class ModelConfig(object):
    device = "cpu"
    gpu_no = "0"
    gpu_options = True
    vocab_size = 10000
    input_embedding_size = 128
    encoder_hidden_units = 64
    decoder_hidden_units = 64
    layer_num = 3
    max_decode_len = 30
    replace_sentence = "I am NLAI"
