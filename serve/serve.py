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
from utils.load_files import LoadDictionary
from seq2seq_.modeling import Seq2SeqModel

index = Blueprint('index', __name__)
api = Blueprint('api', __name__)

con_tf_s = ConnectionTFServing()
load_dict = LoadDictionary()
vd = load_dict.vocab_dict
rvd = load_dict.r_vocab_dict


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
        if request.args:
            input_text = request.args["message"]
            try:
                output_text = Seq2SeqModel.predict_fun(input_text, vd, rvd, con_tf_s)
                return Response(json.dumps({"output_text": output_text}))
            except Exception as e:
                log.error("[predict_fun]" + str(e))
                return Response(json.dumps({"error": "Predict is wrong!"}))
        else:
            return Response(json.dumps({"wain": "No args!"}))
    except Exception as e:
        log.error("[response_info] " + str(e))
