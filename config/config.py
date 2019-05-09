"""
定义模型配置
"""


class ModelConfig(object):
    device = "cpu"
    gpu_no = "0"
    gpu_options = True
    lr = 0.001
    clip_norm = 5
    beam_search = 0
    beam_size = 0
    batch_size = 32
    input_embedding_size = 64
    encoder_hidden_units = 64
    decoder_hidden_units = 64
    layer_num = 3
    decoder_keep_prob = 0.75
    max_decode_len = 30
    replace_sentence = "I am NLAI"
