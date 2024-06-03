import matplotlib.pyplot as plt
import numpy as np
import torch


# 线性回归训练代码
def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / N


def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = torch.tensor(0.0, device=points.device, dtype=torch.float32)
    w_gradient = torch.tensor(0.0, device=points.device, dtype=torch.float32)
    N = float(len(points))
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        b_gradient += -(2 / N) * (y - (w_current * x + b_current))
        w_gradient += -(2 / N) * x * (y - (w_current * x + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]


def gradient_descent_runner(points, starting_b, starting_w, learningRate, num_iterations):
    b = torch.tensor(starting_b, device=points.device, dtype=torch.float32)
    w = torch.tensor(starting_w, device=points.device, dtype=torch.float32)
    for i in range(num_iterations):
        b, w = step_gradient(b, w, points, learningRate)
    return [b, w]


def run():
    points_np = np.genfromtxt("data1.csv", delimiter=',').astype(np.float32)
    points = torch.tensor(points_np, device='mps')
    learning_rate = 0.0001
    initial_b = 0.0
    initial_w = 0.0
    num_iterations = 100000
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After gradient descent at b={0},w={1},error={2}".format(b.item(), w.item(),
                                                                   compute_error_for_line_given_points(b, w, points)))
    return b.item(), w.item()


# 运行线性回归
final_b, final_w = run()

# 绘制图像
points_np = np.genfromtxt("data1.csv", delimiter=',').astype(np.float32)
x = points_np[:, 0]
y = points_np[:, 1]

x_range = np.linspace(min(x), max(x), 100)
y_pred = final_w * x_range + final_b

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Original data')
plt.plot(x_range, y_pred, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Fitting a line to random data')
plt.legend()
plt.grid(True)
plt.savefig('print1.png')
plt.show()
