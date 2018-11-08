# -*- coding:utf-8 -*-
"""
REST API
"""

import json
from flask.app import request, Response
from flask.blueprints import Blueprint
from flask.templating import render_template
from utils.connect import ConnectionTFServing
from utils.loggings import log
from utils.load_files import LoadFiles

index = Blueprint('index', __name__)
api = Blueprint('api', __name__)

con_tf_s = ConnectionTFServing()
load_files = LoadFiles()
vd = load_files.vocab_dict
rvd = load_files.r_vocab_dict


@index.route('/', methods=["GET"])
def main():
    """  主页面

    :return:
    """
    if request.method == 'GET':
        return render_template("index.html")
    else:
        return Response(json.dumps({"WARN": "The request method is wrong!"}))


@api.route('/', methods=['POST', 'GET'])
def response_info():
    """  对话回复

    :return:
    """
    try:
        x = [vd[s] for s in list(request.args["message"])]
        x_data = {"instances": [
            {
                "encoder_inputs": x,
                "decoder_inputs": [0] * len(x)
            },
        ]}
        con_tf_s.calculate_predict_result(x=x_data)
        return "".join([rvd[x]
                        for x in con_tf_s.predict_result["predictions"][0]])
    except Exception as e:
        log.error("[response_info] " + str(e))
