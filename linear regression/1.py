import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt


def run1():
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
                                                                       compute_error_for_line_given_points(b, w,
                                                                                                           points)))

    run()

def run1_cuda():
    def compute_error_for_line_given_points(b, w, points):
        totalError = 0
        N = float(len(points))
        for i in range(len(points)):
            x = points[i][0]
            y = points[i][1]
            totalError += (y - (w * x + b)) ** 2
        return totalError / N

    def step_gradient(b_current, w_current, points, learningRate):
        b_gradient = torch.tensor(0.0, device=points.device)
        w_gradient = torch.tensor(0.0, device=points.device)
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
        b = torch.tensor(starting_b, device=points.device)
        w = torch.tensor(starting_w, device=points.device)
        for i in range(num_iterations):
            b, w = step_gradient(b, w, points, learningRate)
            print("round:", i)
        return [b, w]

    def run():
        points_np = np.genfromtxt("data1.csv", delimiter=',').astype(np.float32)
        points = torch.tensor(points_np, device='cuda')
        learning_rate = 0.0001
        initial_b = 0.0
        initial_w = 0.0
        num_iterations = 100000
        [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
        print("After gradient descent at b={0}, w={1}, error={2}".format(b.item(), w.item(),
                                                                         compute_error_for_line_given_points(b, w,
                                                                                                             points)))
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

def run1x():
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
        num_iterations = 5000
        [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
        print("After gradient descent at b={0},w={1},error={2}".format(b.item(), w.item(),
                                                                       compute_error_for_line_given_points(b, w,
                                                                                                           points)))
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

def run_m1():
    # 检查是否支持MPS（Apple Metal Performance Shaders）
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 生成示例数据
    # y = 3x + 2 + 噪声
    torch.manual_seed(0)
    X = torch.linspace(-10, 10, steps=100).reshape(-1, 1)
    y = 3 * X + 2 + torch.randn(X.size()) * 2

    # 创建数据集和数据加载器
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # 定义线性回归模型
    class LinearRegressionModel(nn.Module):
        def __init__(self):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(1, 1)  # 输入和输出都是1维

        def forward(self, x):
            return self.linear(x)

    # 实例化模型并移动到设备
    model = LinearRegressionModel().to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 保存整个模型
    torch.save(model.state_dict(), 'm1.pth')
    print("整个模型已保存为 m1.pth")

    # 评估模型
    model.eval()
    with torch.no_grad():
        X_test = torch.linspace(-10, 10, steps=100).reshape(-1, 1).to(device)
        y_pred = model(X_test).cpu()

    plt.scatter(X.numpy(), y.numpy(), label='真实数据')
    plt.plot(X_test.cpu().numpy(), y_pred.numpy(), color='red', label='预测线')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('线性回归结果')
    plt.show()

def run_m1_test():
    # 定义线性回归模型结构
    class LinearRegressionModel(nn.Module):
        def __init__(self):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(1, 1)  # 输入和输出都是1维

        def forward(self, x):
            return self.linear(x)

    def main():
        # 检查是否支持MPS（Apple Metal Performance Shaders）
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"使用设备: {device}")

        # 实例化模型并加载保存的模型参数
        model = LinearRegressionModel().to(device)
        model.load_state_dict(torch.load('m1.pth'))
        with open('m1.pth', 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
        print(f"模型文件大小: {size} 字节")
        model.eval()
        # 输出模型大小
        model_size = sum(p.numel() for p in model.parameters())
        print(f"模型大小: {model_size} 个参数")
        print("模型参数已加载")

        # 生成测试数据
        X_test = torch.linspace(-10, 10, steps=100).reshape(-1, 1).to(device)

        # 使用加载的模型进行预测
        with torch.no_grad():
            y_pred = model(X_test).cpu()

        # 将测试数据移至CPU并转换为NumPy数组
        X_test_numpy = X_test.cpu().numpy()
        y_pred_numpy = y_pred.numpy()

        # 可视化预测结果
        plt.scatter(X_test_numpy, 3 * X_test_numpy + 2, label='真实线性关系', color='blue')
        plt.plot(X_test_numpy, y_pred_numpy, color='red', label='模型预测线')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('加载模型后的线性回归预测结果')
        plt.show()

    main()

if __name__ == '__main__':
    print("start")
