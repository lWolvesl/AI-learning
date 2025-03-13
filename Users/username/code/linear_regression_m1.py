import torch

# 检查MPS可用性（需要PyTorch 1.12+和macOS 12.3+）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 生成训练数据（移动到MPS设备）
X = torch.randn(1000, 2).to(device)  # 1000个样本，2个特征
y = X @ torch.tensor([2.0, -3.4], device=device) + 4  # 真实关系式
y += 0.01 * torch.randn(y.shape, device=device)  # 添加噪声

# 定义模型（必须继承nn.Module）
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)  # 输入2维，输出1维
        
    def forward(self, x):
        return self.linear(x)

model = LinearRegression().to(device)  # 将模型移至MPS设备
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 训练循环
for epoch in range(500):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y.unsqueeze(1))
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, loss: {loss.item():.4f}')

# 输出最终参数
print("Learned weights:", model.linear.weight.data)
print("Learned bias:", model.linear.bias.data)
