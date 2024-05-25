import numpy as np

a1 = np.array([1, 2, 3, 4, 5])

print(a1)

a2 = np.zeros((2, 3), dtype=int)

print(a2)

print(a1.shape)
print(a2.shape)

a3 = np.ones((2, 4), dtype=int)

print(a3)

a4 = np.arange(1, 10)

print(a4)

a5 = np.linspace(1, 6, 4)

print(a5)

a6 = np.random.rand(2, 4)

print("a6: " + a6.__str__())

a7 = a5.astype(int)

print(a7)

a8 = np.linspace(1, 10, 4).astype(int)
a9 = np.linspace(1, 6, 4).astype(int)

print(a8)
print(a9)

a10 = np.dot(a8, a9)

print(a10)

a11 = np.zeros((2, 3), dtype=int)

a11[0] = np.linspace(1, 10, 3).astype(int)
a11[1] = np.linspace(1, 11, 3).astype(int)

print(a11)

# numpy数组可以直接进行线性变换 即广播
a12 = (1 + (10 - 1) * np.random.rand(2, 2)).astype(int)
a13 = (1 + (10 - 1) * np.random.rand(2, 2)).astype(int)

print("a12: " + a12.__str__())
print("a13: " + a13.__str__())

# 矩阵乘法 写法等同
a14 = a12 @ a13
print(a14)

a14 = np.dot(a12, a13)
print(a14)

a14 = np.matmul(a12, a13)
print(a14)

a15 = np.sqrt(a12)
print(a15)

a16 = np.sin(a12)
a17 = np.cos(a12)

print(a16)
print(a17)

a18 = np.log(a12)
print(a18)

a19 = np.power(a12, 2)
print(a19)

ax1 = (1 + (10 - 1) * np.random.rand(1, 7)).astype(int)
print("ax1: " + ax1.__str__())

a20 = np.max(np.abs(ax1))
a21 = np.min(np.abs(ax1))
print(a20)
print(a21)

a22 = np.argmax(np.abs(ax1))
a23 = np.argmin(np.abs(ax1))
print(a22)
print(a23)

a24 = np.sum(np.abs(ax1))
print(a24)

a25 = np.mean(np.abs(ax1))
print(a25)

a26 = np.median(np.abs(ax1))
print(a26)

# 方差
a27 = ax1.var()
print(a27)

a28 = ax1.std()
print(a28)

a29 = ax1[(ax1 > 3) & (ax1 % 2 == 0)]
print(a29)

a30 = ax1[0, 0:6:2]
print(a30)

# 反转数组
a31 = ax1[::-1]
print(a31)
