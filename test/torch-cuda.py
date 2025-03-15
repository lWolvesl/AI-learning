import torch

# 打印 PyTorch 版本
print("PyTorch 版本:", torch.__version__)

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA 可用: {cuda_available}")

# 如果 CUDA 可用，打印 GPU 信息
if cuda_available:
    print("\nGPU 详细信息：")
    print("GPU 数量:", torch.cuda.device_count())
    print("当前 GPU:", torch.cuda.current_device())
    print("GPU 名称:", torch.cuda.get_device_name(0))
    print("GPU 内存:")
    print("  已分配:", torch.cuda.memory_allocated())
    print("  保留:", torch.cuda.memory_reserved())
else:
    print("\n未检测到GPU设备")