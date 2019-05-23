## 智能对话机器人
#### 环境说明
* python环境: python-3.5
* 依赖包见 requirements.txt

#### 执行说明
* step-1 训练模型，语料在corpus文件下，语料可替换，格式要与示例语料相同

 ```bash
python run_train.py --model_version 001 --train_steps_num  # 指定模型版本和迭代轮次
 ```

* step-2 启动机器人，访问 http://127.0.0.1:5000 测试界面

  * 测试启动
  
   ```bash
   python run_robot.py
   ```
  
  * 正式启动
  
  ```bash
  gunicorn -w 4 -b 0.0.0.0:5000 run_robot:app  # 适用于linux系统下启动，win下会报错ModuleNotFoundError: No module named 'pwd'
  ```
  
#### 部署机器人到微信

```bash
python run_wei_chat.py
```

#### 注意说明
 
  * 训练过程中，若要将模型部署到 tensorflow-serving 中，令show_text=False，因为 tensorflow-serving 框架中貌似不支持py_func函数，会导致部署模型时图加载失败。
 
  * 若在启动 run_tf_serving.py 时候，出现<font color="red">Error starting userland proxy: mkdir /port/tcp:0.0.0.0:8501:tcp:172.17.0.2:8501: input/output error</font>, 重启docker问题可解决。
  
  * 启动界面访问url使用 https开头, 则会出现 code 400, message Bad request version， 改http即可。
  
  * 在win系统下，二维码生成到项目目录下; 在linux系统下，二维码打印在控制台，扫码即可登录。
  
  * 在win10下安装pyltp，不要采用pip install pyltp，因为编译会不通过，可参考 [https://blog.csdn.net/weixin_40899194/article/details/79702468](https://blog.csdn.net/weixin_40899194/article/details/79702468) 方式进行安装。
  
  * 在采用beam_search模式进行解码时，会遇到KeyError: 'GatherTree'的问题，则在加载模型时加入 from tensorflow.contrib.seq2seq.python.ops import beam_search_ops可解决此问题，具体原因见 [https://github.com/tensorflow/tensorflow/issues/12927](https://github.com/tensorflow/tensorflow/issues/12927)
  
  * 在调用百度语音识别API时，需要安装ffmseg，并配置好环境变量。
  
#### 未解决问题
  *  通过tf-serving请求模型预测在beam_search模式下，不能得到结果。
 