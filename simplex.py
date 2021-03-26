import numpy as np

M = 999999


def simplex(A, b, c, base, n):
    S = np.concatenate([A, b], 1)
    Z = np.concatenate([c[:, 0], [0]], 0)
    while 1:
        # 选择入基出基变量、计算θ
        max_c = max(Z[:-1])
        if max_c <= 0:
            break
        in_base = np.argmax(Z[:-1])  # 列
        theta = [row[-1] / row[in_base] if row[in_base] >= 0 else M for row in S]
        # theta = [i if i >= 0 else M for i in theta]
        if min(theta) == M:
            print("最优解为无穷大")
            return None
        out_base = np.argmin(theta)  # 行

        # 进行基的替换
        base[out_base] = in_base
        scale = 1 / S[out_base][in_base]
        S[out_base] *= scale
        for i in range(S.shape[0]):
            if i == out_base:
                continue
            scale = S[i][in_base] * -1
            S[i] += S[out_base] * scale

        # 更新-Z
        scale = Z[in_base] * -1
        Z += S[out_base] * scale

        # 输出这一步的结果
        # print(S)
        # print(Z)
    x = np.zeros(n)
    for i in range(n):
        if i in base:
            x[i] = S[np.where(base == i)][0][-1]
    return -Z[-1], x


# 测试
n1 = 3
A1 = np.array([[1, 4, 2, 1, 0],
              [1, 2, 4, 0, 1]], dtype='float64')
b1 = np.array([[48], [60]], dtype='float64')
c1 = np.array([[6], [14], [13], [0], [0]], dtype='float64')
base1 = np.array([3, 4])
Z1, x1 = simplex(A1, b1, c1, base1, n1)
print("第一题(9a)最优解为Z={},x={}".format(Z1, x1))

n2 = 3
A2 = np.array([[4, 5, -2, 1, 0],
              [1, -2, 1, 0, 1]], dtype='float64')
b2 = np.array([[22], [30]], dtype='float64')
c2 = np.array([[-3], [2], [4], [0], [0]], dtype='float64')
base2 = np.array([3, 4])
Z2, x2 = simplex(A2, b2, c2, base2, n2)
print("第二题(9b)最优解为Z={},x={}".format(Z2, x2))

n3 = 3
A3 = np.array([[2, -1, 3, 1, 0],
              [1, 2, 4, 0, 1]], dtype='float64')
b3 = np.array([[30], [40]], dtype='float64')
c3 = np.array([[4], [2], [8], [0], [-M]], dtype='float64')
base3 = np.array([3, 4])
Z3, x3 = simplex(A3, b3, c3, base3, n3)
print("第三题(10)最优解为Z={},x={}".format(Z3, x3))

