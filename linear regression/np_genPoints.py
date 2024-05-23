import numpy as np
import csv

# 定义回归方程参数
w = 1.35
b = 2.89

# 生成x值范围
x_min = 0
x_max = 10

# 生成100个在x轴附近的点
x = np.linspace(x_min, x_max, 100)

# 根据回归方程计算y值
y = w * x + b

# 添加一些噪声，使数据更真实
y += np.random.normal(scale=0.5, size=y.shape)

# 将x和y合并成一个二维数组
data = np.column_stack((x, y))

# 将数据保存到CSV文件
with open('data1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 写入表头
    # writer.writerow(['x', 'y'])
    # 写入数据
    writer.writerows(data)
