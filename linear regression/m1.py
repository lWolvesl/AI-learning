import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


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
