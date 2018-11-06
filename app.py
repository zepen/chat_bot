# -*- coding:utf-8 -*-
"""
启动服务
"""

from flask import Flask
from serve.serve import index, api

app = Flask(__name__)
app.register_blueprint(index, url_prefix='/')
app.register_blueprint(api, url_prefix='/api')
print("[INFO] chat bot is running!")


if __name__ == '__main__':
    app.run()
