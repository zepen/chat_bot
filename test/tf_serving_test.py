# -*- coding: utf-8 -*-
"""
测试tf_serving是否通畅
"""
import os
from utils.connect import ConnectionTFServing

os.chdir("..")
con_tf_s = ConnectionTFServing()
print("URL {}".format(con_tf_s.url))

data = {
        "instances": [
           {
               "encoder_inputs": [0],
               "decoder_inputs": [1]
           },
        ]
       }

if __name__ == '__main__':
    con_tf_s.calculate_predict_result(data)
    print(con_tf_s.predict_result)
