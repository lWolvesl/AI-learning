import numpy as np
import torch


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
        b_gradient += -(2 / N) * (y - (w_current * x + b_current) + b_current)
        w_gradient += -(2 / N) * x * (y - (w_current * x + b_current + b_current))
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
    # 修改为生成数据的文件路径
    points_np = np.genfromtxt("data1.csv", delimiter=',').astype(np.float32)
    points = torch.tensor(points_np, device='mps')
    learning_rate = 0.0001  # 使用较小的学习率
    initial_b = 0.0
    initial_w = 0.0
    num_iterations = 1000
    print("Starting gradient descent at b={0},w={1},error={2}".format(initial_b, initial_w,
                                                                      compute_error_for_line_given_points(initial_b,
                                                                                                          initial_w,
                                                                                                          points)))
    print("running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After gradient descent at b={0},w={1},error={2}".format(b.item(), w.item(),
                                                                   compute_error_for_line_given_points(b, w, points)))


if __name__ == '__main__':
    run()
