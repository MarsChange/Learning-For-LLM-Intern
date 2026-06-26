# 手撕 KMeans（NumPy）

## 面试定位

KMeans 常考的是算法过程，不允许直接调 `scikit-learn`。这里严格只使用 NumPy：

1. 初始化中心：取前 `k` 个样本。
2. 计算每个点到每个中心的平方欧氏距离。
3. 分配 label。
4. 按 label 重新计算中心。
5. label 或 center 不再变化时停止。

空簇处理：如果某个簇没有样本，中心保持不变。

## 输入输出格式

输入：

```text
n d k max_iter
x_1 的 d 维特征
...
x_n 的 d 维特征
```

输出：

```text
n 个 label
k 行中心，每行 d 个数
```

距离相同的情况，`np.argmin` 会选编号更小的簇。中心保留 6 位小数。

## 可运行代码

```python
import sys
import numpy as np


def fmt(x: float) -> str:
    # 输出中心点坐标时保留 6 位小数，极小误差当作 0。
    if abs(x) < 5e-7:
        x = 0.0
    return f"{x:.6f}"


def kmeans(points, k, max_iter):
    # points: [n, d]，n 个样本，每个样本 d 维。
    n, d = points.shape
    # 为了方便判题，初始化策略固定为“前 k 个样本作为中心”。
    centers = points[:k].astype(float).copy()
    # 初始 label 设为 -1，保证第一轮一定会更新。
    labels = np.full(n, -1, dtype=int)

    for _ in range(max_iter):
        # 广播计算所有点到所有中心的差：
        # points[:, None, :] -> [n,1,d]
        # centers[None, :, :] -> [1,k,d]
        # diff -> [n,k,d]
        diff = points[:, None, :] - centers[None, :, :]
        # 平方欧氏距离，dist2[i,c] 表示第 i 个点到第 c 个中心的距离。
        dist2 = np.sum(diff * diff, axis=2)
        # 每个点分配给距离最近的中心；距离相同时 argmin 选编号更小的簇。
        new_labels = np.argmin(dist2, axis=1)

        # 先复制旧中心；如果某个簇为空，就保持旧中心不变。
        new_centers = centers.copy()
        for c in range(k):
            # 取出当前簇 c 的所有成员点。
            members = points[new_labels == c]
            if len(members) > 0:
                # 新中心是该簇所有点在每个维度上的均值。
                new_centers[c] = members.mean(axis=0)

        # 收敛条件：label 不变，或中心几乎不变。
        if np.array_equal(new_labels, labels) or np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break

        # 进入下一轮迭代。
        labels = new_labels
        centers = new_centers

    return labels, centers


def solve():
    # 读取样本数 n、维度 d、簇数 k、最大迭代次数。
    data = sys.stdin.read().strip().split()
    if not data:
        return

    n, d, k, max_iter = map(int, data[:4])
    vals = np.array(list(map(float, data[4:])), dtype=float)
    # 恢复样本矩阵 [n,d]。
    points = vals.reshape(n, d)

    labels, centers = kmeans(points, k, max_iter)
    # 先输出每个样本的簇编号，再输出最终中心。
    print(" ".join(map(str, labels.tolist())))
    for row in centers:
        print(" ".join(fmt(x) for x in row))


if __name__ == "__main__":
    solve()
```

## 纯 Python 标准库版本

如果面试明确要求“不使用 NumPy”，可以写下面这个版本。输入输出和上面的 NumPy 版本完全一致，只使用 Python 基本语法和 `sys`。

核心思路不变：

- 用二重循环计算每个点到每个中心的距离。
- 用列表保存 `labels` 和 `centers`。
- 更新中心时手动累加每个簇的坐标和样本数。
- 空簇保持旧中心不变。

```python
import sys


def fmt(x: float) -> str:
    # 输出中心点坐标时保留 6 位小数，极小误差当作 0。
    if abs(x) < 5e-7:
        x = 0.0
    return f"{x:.6f}"


def squared_distance(a, b):
    # 计算两个 d 维点的平方欧氏距离：
    # sum_j (a_j - b_j)^2。
    dist = 0.0
    for x, y in zip(a, b):
        diff = x - y
        dist += diff * diff
    return dist


def centers_close(a, b):
    # 对齐 NumPy allclose 的常见容差逻辑。
    # 中心几乎不变时，可以认为算法已经收敛。
    for row_a, row_b in zip(a, b):
        for x, y in zip(row_a, row_b):
            if abs(x - y) > 1e-8 + 1e-5 * abs(y):
                return False
    return True


def kmeans(points, k, max_iter):
    # points: List[List[float]]，形状可以理解为 [n, d]。
    n = len(points)
    d = len(points[0])

    # 初始化策略固定为“前 k 个点作为中心”，便于面试和 ACM 判题复现。
    centers = [point[:] for point in points[:k]]
    # 初始 label 设为 -1，保证第一轮一定会更新。
    labels = [-1] * n

    for _ in range(max_iter):
        new_labels = []

        # 1. 分配步骤：对每个点，找到距离最近的中心。
        for point in points:
            best_cluster = 0
            best_dist = squared_distance(point, centers[0])

            # 从第 1 个中心开始比较；只在严格更小时更新。
            # 如果距离相同，会保留编号更小的中心，等价于 np.argmin 的行为。
            for c in range(1, k):
                dist = squared_distance(point, centers[c])
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = c

            new_labels.append(best_cluster)

        # 2. 更新步骤：先准备每个簇的坐标和与样本数量。
        sums = [[0.0 for _ in range(d)] for _ in range(k)]
        counts = [0 for _ in range(k)]

        for point, label in zip(points, new_labels):
            counts[label] += 1
            for j in range(d):
                sums[label][j] += point[j]

        # 先复制旧中心。遇到空簇时，不更新该中心。
        new_centers = [center[:] for center in centers]
        for c in range(k):
            if counts[c] > 0:
                for j in range(d):
                    new_centers[c][j] = sums[c][j] / counts[c]

        # 3. 收敛判断：label 不变，或中心几乎不变。
        if centers_close(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break

        # 进入下一轮迭代。
        labels = new_labels
        centers = new_centers

    return labels, centers


def solve():
    # 输入格式：
    # n d k max_iter
    # 后面跟 n*d 个浮点数，表示 n 个 d 维点。
    data = sys.stdin.read().strip().split()
    if not data:
        return

    n, d, k, max_iter = map(int, data[:4])
    vals = list(map(float, data[4:]))

    # 把扁平输入恢复成 points[i][j]。
    points = []
    idx = 0
    for _ in range(n):
        points.append(vals[idx:idx + d])
        idx += d

    labels, centers = kmeans(points, k, max_iter)

    # 输出格式与 NumPy 版本保持一致。
    print(" ".join(str(label) for label in labels))
    for center in centers:
        print(" ".join(fmt(x) for x in center))


if __name__ == "__main__":
    solve()
```

## 测试用例

### 用例 1

输入：

```text
4 2 2 10
0 0
0 2
10 10
10 12
```

输出：

```text
0 0 1 1
0.000000 1.000000
10.000000 11.000000
```

### 用例 2

输入：

```text
3 1 2 10
0
10
11
```

输出：

```text
0 1 1
0.000000
10.500000
```

### 用例 3

输入：

```text
5 2 2 10
0 0
0 1
1 0
9 9
10 9
```

输出：

```text
0 0 0 1 1
0.333333 0.333333
9.500000 9.000000
```

## 易错点

- 不要调用 `sklearn.cluster.KMeans`。
- 距离矩阵可以广播成 `[n,k,d]`，再对最后一维求和。
- 中心更新时要处理空簇。
- 初始化策略会影响最终结果；这里固定“前 k 个点”为中心，方便 ACM 判题。
