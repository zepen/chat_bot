# -*- coding:utf-8 -*-
"""
FLASK RESTFUL API
"""

import json
from flask.app import request, Response
from flask.blueprints import Blueprint
from flask.templating import render_template
from algorithm.ner import Ltp
from algorithm.language_model import BaiduDnnLM
from utils.loggings import log
from utils.load_files import RuleCorrection
from .docker_serve import predict_func

index = Blueprint('index', __name__)
api = Blueprint('api', __name__)


ltp = Ltp()
baidu_dnn_lm = BaiduDnnLM()
rule_c = RuleCorrection()


def information_retrieval(entity):
    """ 信息检索

    :param entity: 获取实体
    :return:
    """
    if entity:
        return "我认识" + entity


def generate_response(input_text):
    """ 生成回复

    :param input_text: 输入当前句
    :return:
    """
    response = predict_func(input_text)
    print(response)
    if len(response):
        response_ppl = baidu_dnn_lm(text=response)['ppl']
        print(response_ppl)
        if response_ppl < 500:
            return rule_c.replace_name(response)
        else:
            return rule_c.replace_content()
    return rule_c.replace_content()


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
            entity = ltp.get_entity(input_text)
            if entity:
                return information_retrieval(entity)
            input_text_ppl = baidu_dnn_lm(text=input_text)['ppl']
            if input_text_ppl < 500:
                return generate_response(input_text)
            else:
                return rule_c.replace_content()
        else:
            return "What?"
    except Exception as e:
        log.error("[response_info] " + str(e))
        return "ERROR"
