import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    main()
