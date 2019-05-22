# -*- coding:utf-8 -*-
"""
启动robot
"""
from flask import Flask
from serve.docker_serve import docker_run
from serve.flask_serve import index, api


docker_run("docker/")
app = Flask(__name__)
app.register_blueprint(index)
app.register_blueprint(api)
print("[INFO] chat bot is running!")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
