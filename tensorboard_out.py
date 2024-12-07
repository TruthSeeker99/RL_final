import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 指定 TensorBoard 日志目录
log_dir = "./logs"

# 加载日志文件
event_acc = EventAccumulator(log_dir, size_guidance={'scalars': 1000})
event_acc.Reload()

# 获取特定标签的数据（如 "loss"）
tags = event_acc.Tags()['scalars']
print("Available tags:", tags)

scalar_name = "loss"  # 替换为你的标量名称
steps = []
values = []

for event in event_acc.Scalars(scalar_name):
    steps.append(event.step)
    values.append(event.value)

# 绘制图像
plt.plot(steps, values)
plt.xlabel("Step")
plt.ylabel(scalar_name)
plt.title(f"TensorBoard Scalar: {scalar_name}")
plt.grid()
plt.savefig(f"{scalar_name}.png")  # 保存为图像
plt.show()
