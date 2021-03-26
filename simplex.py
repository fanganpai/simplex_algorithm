import numpy as np

M = 999999

# 初始化数据
n = 3
A = np.array([[1, 4, 2, 1, 0],
              [1, 2, 4, 0, 1]], dtype='float64')
b = np.array([[48], [60]], dtype='float64')
c = np.array([[6], [14], [13], [0], [0]], dtype='float64')
base = np.array([3, 4])


# 单纯形法
S = np.concatenate([A, b], 1)
Z = np.concatenate([c[:, 0], [0]], 0)
while 1:
    # 选择入基出基变量、计算θ
    max_c = max(Z[:-1])
    if max_c <= 0:
        break
    in_base = np.argmax(Z[:-1])  # 列
    theta = [row[-1] / row[in_base] for row in S]
    theta = [i if i >= 0 else M for i in theta]
    if min(theta) == M:
        print("最优解为无穷大")
        break
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
    print(S)
    print(Z)
x = np.zeros(n)
for i in range(n):
    if i in base:
        x[i] = S[np.where(base == i)][0][-1]
print("最优解为Z={},x={}".format(Z[-1], x))
