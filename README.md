## 智能对话机器人
#### 环境说明
* python-3.5
* tensorflow-1.10.1

#### 执行说明
* step-1 训练模型，语料在corpus文件下，语料可替换，格式要与示例语料相同

 ```
python run_train.py --model_version 001  # 指定模型版本
 ```

* step-2 启动 tensorflow_serving docker 服务

```bash
python run_tf_serving.py
```

* step-3 启动对话机器人界面
  * 测试启动
   ```bash
   python run_robot.py
   ```
  * 正式启动
  ```bash
  gunicorn -w 4 -b 0.0.0.0:5000 run_robot:app
  ```

 #### 注意说明
  若在启动 run_tf_serving.py 时候，
  出现<font color="red">Error starting userland proxy: mkdir /port/tcp:0.0.0.0:8501:tcp:172.17.0.2:8501: input/output error</font>
  重启docker问题可解决