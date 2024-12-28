import numpy as np

# 创建一个二维数组
array = np.array([[1, 2, 3], [4, 5, 6]])

# 对整个数组求和
total_sum = np.sum(array)

# 对每一列求和
column_sum = np.sum(array, axis=0)

# 对每一行求和
row_sum = np.sum(array, axis=1)

print("总和:", total_sum)
print("列和:", column_sum)
print("行和:", row_sum)