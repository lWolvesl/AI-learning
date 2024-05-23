import numpy as np
import matplotlib.pyplot as plt

# 原始数据
points = np.genfromtxt("data1.csv", delimiter=',')
x = points[:, 0]
y = points[:, 1]

# 拟合直线
x_range = np.linspace(min(x), max(x), 100)
y_pred = 1.6455038785934448 * x_range + 1.827562689781189

# 绘图
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Original data')
plt.plot(x_range, y_pred, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Fitting a line to random data')
plt.legend()
plt.grid(True)
plt.savefig('print1.png')
