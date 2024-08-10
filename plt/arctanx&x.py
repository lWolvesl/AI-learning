import numpy as np
import matplotlib.pyplot as plt

# 定义 x 的范围
x = np.linspace(-3, 3, 400)

# 计算 y 的值
y_arctan = np.arctan(x)
y_linear = x

# 绘制图形
plt.figure(figsize=(8, 6))
plt.plot(x, y_arctan, label='y = arctan(x)', color='blue')
plt.plot(x, y_linear, label='y = x', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y = arctan(x) and y = x')

# 获取当前坐标轴
ax = plt.gca()

# 设置 x 轴和 y 轴的线条粗细
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

plt.legend()
plt.grid(True)  # 如果需要网格线，可以保留这行
plt.savefig('arctanx&x.png')
plt.show()
