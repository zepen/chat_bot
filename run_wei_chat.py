# -*- coding:utf-8 -*-
"""
启动在微信后台，生成回复
"""
import sys
import json
import itchat
import requests as req

# 设置QR值，会在控制台生成二维码
if sys.platform == "win32":
    QR = False
elif sys.platform == "linux":
    QR = 2
else:
    QR = 1


@itchat.msg_register(itchat.content.TEXT)
def reply_msg(msg):
    info = msg['Content']
    # 机器人API
    api_url = 'http://localhost:5000/api'
    # 请求接口
    result = req.post(api_url, data=json.dumps({"content": info}))
    print("[INFO][code: {}][message: {}]".format(result.status_code, result.text))
    # 提取text，发送给发信息的人
    itchat.send_msg(result.text, msg['FromUserName'])


if __name__ == '__main__':
    itchat.auto_login(enableCmdQR=QR)
    itchat.run()
