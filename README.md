## 智能对话机器人
#### 环境说明


#### 执行说明
 启动 tensorflow_serving docker 服务
```bash
python run_tf_serving.py
```
 启动对话机器人界面
```bash
python run_robot.py
```
 训练模型
 ```
python run_train.py
 ```
 
 #### 注意说明
  若在启动 run_tf_serving.py 时候，出现Error starting userland proxy: mkdir /port/tcp:0.0.0.0:8501:tcp:172.17.0.2:8501: input/output error，
  重启docker可解决