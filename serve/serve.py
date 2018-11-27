# -*- coding:utf-8 -*-
"""
REST API
"""

import json
from flask.app import request, Response
from flask.blueprints import Blueprint
from flask.templating import render_template
from algorithm.seq2seq_.modeling import Seq2SeqModel
from algorithm.classifier.modeling import ClassifierModel
from utils.connect import ConnectionTFServing
from utils.loggings import log
from utils.load_files import LoadDictionary
from utils.model_config import ModelConfig

index = Blueprint('index', __name__)
api = Blueprint('api', __name__)

seq2seq = Seq2SeqModel()
classifier = ClassifierModel()
con_tf_s = ConnectionTFServing()
m_config = ModelConfig()
load_files = LoadDictionary()
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


@api.route('/api', methods=['POST'])
def response_info():
    """  对话回复

    :return: str
    """
    try:
        if request.data:
            request_text = request.data.decode("utf-8")
            request_text = request_text.replace("\n", "")
            input_text = json.loads(request_text)["content"]
            # classifier_res = ClassifierModel.predict_fun()
            # if classifier_res:
            #     pass
            return Seq2SeqModel.predict_fun(input_text, vd, rvd, con_tf_s, m_config)
        else:
            return "What?"
    except Exception as e:
        log.error("[response_info] " + str(e))
        return "ERROR"
