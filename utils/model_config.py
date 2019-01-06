"""
定义模型配置
"""


class ModelConfig(object):
    device = "cpu"
    gpu_no = "0"
    gpu_options = True
    vocab_size = None
    input_embedding_size = 128
    encoder_hidden_units = 64
    decoder_hidden_units = 64
    max_decode_len = 30
    replace_sentence = "I am NLAI"
