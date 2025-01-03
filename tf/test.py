import tensorflow as tf

# 创建一个TensorFlow常量
hello = tf.constant('Hello, TensorFlow!')

# 将TensorFlow张量转换为NumPy数组
numpy_array = hello.numpy()

print(numpy_array)  # 输出: b'Hello, TensorFlow!'
