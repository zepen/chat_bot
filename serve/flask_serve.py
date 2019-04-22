# -*- coding:utf-8 -*-
"""
FLASK RESTFUL API
"""

import json
from flask.app import request, Response
from flask.blueprints import Blueprint
from flask.templating import render_template
from algorithm.processing import RuleCorrection
from utils.loggings import log
from utils.load_files import LoadDictionary
from config.config import ModelConfig
from .tf_serving import predict_func

index = Blueprint('index', __name__)
api = Blueprint('api', __name__)

load_files = LoadDictionary()
vd = load_files.vocab_dict
rvd = load_files.r_vocab_dict
vocab_size = len(vd)
m_config = ModelConfig()
rule_c = RuleCorrection()


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
            return predict_func(input_text, vd, rvd, m_config, rule_c)
        else:
            return "What?"
    except Exception as e:
        log.error("[response_info] " + str(e))
        return "ERROR"
