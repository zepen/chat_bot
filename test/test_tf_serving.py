# -*- coding: utf-8 -*-
"""
采用locust测试对服务进行压力测试
pip install locust
locust -f test_tf_serving
"""
from locust import HttpLocust, TaskSet, task


class UserTasks(TaskSet):

    @task
    def get_r(self):
        url = '/v1/models/chat_bot:predict'
        data = {
            "instances": [
                {
                    "encoder_inputs": list("你是谁"),
                    "encoder_inputs_length": 3,
                    "batch_size": 1
                },
            ]
        }
        self.client.post(url, json=data)


class WebsiteUser(HttpLocust):
    host = "http://127.0.0.1:8501"
    min_wait = 2000
    max_wait = 5000
    task_set = UserTasks
