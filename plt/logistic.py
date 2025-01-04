import numpy as np
import matplotlib.pyplot as plt

def logistic_function(x, w, b):
    return 1 / (1 + np.exp(-(w * x + b)))

def cost_function(f_wb, y):
    if y == 1:
        return -np.log(f_wb)
    else:
        return -np.log(1 - f_wb)

f_wb = np.linspace(-1, 2, 100)  # 避免log(0)的情况

cost_y1 = cost_function(f_wb, 1)
cost_y0 = cost_function(f_wb, 0)

plt.figure(figsize=(12, 6),dpi=600)

plt.subplot(1, 2, 1)
plt.plot(f_wb, cost_y1, label='y=1')
plt.title('Cost Function when y=1')
plt.xlabel('f_wb')
plt.ylabel('Cost')
plt.axhline(0, color='black', linewidth=0.8)  # 增加水平坐标轴
plt.axvline(0, color='black', linewidth=0.8)  # 增加垂直坐标轴
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(f_wb, cost_y0, label='y=0')
plt.title('Cost Function when y=0')
plt.xlabel('f_wb')
plt.ylabel('Cost')
plt.axhline(0, color='black', linewidth=0.8)  # 增加水平坐标轴
plt.axvline(0, color='black', linewidth=0.8)  # 增加垂直坐标轴
plt.legend()

plt.tight_layout()
#plt.show()
plt.savefig('plt/logistic_cost_function.png')