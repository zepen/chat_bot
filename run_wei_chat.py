# -*- coding:utf-8 -*-
"""
启动在微信后台，生成回复
"""
import os
import sys
import json
import itchat
import numpy as np
import requests as req
from pydub import AudioSegment
from algorithm.speech_recognizer import BaiduSpeechRecognizer

# 机器人API
api_url = 'http://localhost:5000/api'
sr = BaiduSpeechRecognizer()

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
    # 请求接口
    result = req.post(api_url, data=json.dumps({"content": info}))
    print("[INFO][code: {}][message: {}]".format(result.status_code, result.text))
    # 提取text，发送给发信息的人
    itchat.send_msg(result.text, msg['FromUserName'])


@itchat.msg_register(itchat.content.VOICE)
def reply_msg(voice):
    voice_name = voice["FileName"]
    # 下载语音文件
    voice['Text'](voice_name)
    wav_voice_name = voice_name.split(".")[0]
    if os.path.exists(voice_name):
        song = AudioSegment.from_mp3(voice_name)
        song.export(wav_voice_name + ".wav", format="wav")
        with open(wav_voice_name + ".wav", 'rb') as f:
            voice_data = f.read()
        reply = {}
        try:
            reply = sr(voice=voice_data)
            os.remove(voice_name)
            os.remove(wav_voice_name + ".wav")
        except Exception as e:
            print(e)
            reply["err_no"] = 9999
        print(reply)
        if reply["err_no"] == 0:
            info = reply["result"][0]
            result = req.post(api_url, data=json.dumps({"content": info}))
            print("[INFO][code: {}][message: {}]".format(result.status_code, result.text))
            # 提取text，发送给发信息的人
            itchat.send_msg(result.text, voice['FromUserName'])
        else:
            random = np.random.randint(len(sr.voice_reply))
            print("[INFO][baidu_code: {}][message: {}]".format(reply["err_no"], sr.voice_reply[random]))
            itchat.send_msg(sr.voice_reply[random], voice['FromUserName'])


@itchat.msg_register(itchat.content.PICTURE)
def reply_msg(picture):
    print(picture['Content'])


if __name__ == '__main__':
    itchat.auto_login(enableCmdQR=QR)
    itchat.run()
