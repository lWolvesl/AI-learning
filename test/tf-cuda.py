import tensorflow as tf

# 打印 TensorFlow 版本
print("TensorFlow 版本:", tf.__version__)

# 检查 CUDA 是否可用
cuda_available = tf.test.is_built_with_cuda()
print(f"CUDA 支持: {cuda_available}")

# 检查 GPU 是否可用
gpu_available = tf.config.list_physical_devices('GPU')
print(f"GPU 可用: {gpu_available}")

# 打印可用设备
print("\n可用设备：")
for device in tf.config.list_physical_devices():
    print(device)

# 如果 GPU 可用，打印详细信息
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("\nGPU 详细信息：")
    for gpu in gpus:
        print(gpu)
        print("设备名称:", gpu.name)
        print("设备类型:", gpu.device_type)
else:
    print("\n未检测到GPU设备")