# -*- coding:utf-8 -*-
"""
启动服务
"""
from flask import Flask
from serve.serve import index, api

app = Flask(__name__)
app.register_blueprint(index)
app.register_blueprint(api)
print("[INFO] chat bot is running!")


if __name__ == '__main__':
    app.run()
