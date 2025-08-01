import numpy as np

# 定义系数矩阵A和常数向量b
A = np.array([[2, 1], [1, 3]])
b = np.array([4, 5])

# 解方程组
x = np.linalg.solve(A, b)
print(x)  # 输出解向量

F = np.concatenate([[1, 2, 3], [3, 2, 1]])
print(F)

j11 = [111, 112, 113, 114]
j12 = [221, 222, 223, 224]
j21 = [331, 332, 333, 334]
j22 = [441, 442, 443, 444]

J = np.block([
    [j11, j12],
    [j21, j22]
])
print(J)

X = np.array([[1, 1, 1],
              [2, 2, 2],
              [3, 3, 3]])

W = np.array([[2, 0, 0],
              [0, 3, 0],
              [0, 0, 4]])

print(X @ W)

print(X.shape)
print(2 ** 2)

# d_y1 = 905 - x_1**2 - x_2
# d_y2 = 55 - x_1 - x_2**2
d_y1 = 0
d_y2 = 0
x_1 = 10
x_2 = 10
for i in range(10):
    y1 = x_1**2 + x_2
    y2 = x_1 + x_2**2
    d_y1 = 905 - y1
    d_y2 = 55 - y2
    d_y = [d_y1, d_y2]
    print("迭代次数:", i, "x1 =", x_1, "x2 =", x_2, "误差值:", max(abs(d_y1), abs(d_y2)))
    if max(abs(d_y1), abs(d_y2)) < 1e-8:
        break
    H = -2 * x_1
    N = -1
    M = -1
    L = -2 * x_2
    J = np.block([
        [H, N],
        [M, L]
    ])
    dx = np.linalg.solve(-J, d_y)
    x_1 += dx[0]
    x_2 += dx[1]

print(np.cos(0))
