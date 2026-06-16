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
    if abs(x) < 5e-7:
        x = 0.0
    return f"{x:.6f}"


def kmeans(points, k, max_iter):
    n, d = points.shape
    centers = points[:k].astype(float).copy()
    labels = np.full(n, -1, dtype=int)

    for _ in range(max_iter):
        diff = points[:, None, :] - centers[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        new_labels = np.argmin(dist2, axis=1)

        new_centers = centers.copy()
        for c in range(k):
            members = points[new_labels == c]
            if len(members) > 0:
                new_centers[c] = members.mean(axis=0)

        if np.array_equal(new_labels, labels) or np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break

        labels = new_labels
        centers = new_centers

    return labels, centers


def solve():
    data = sys.stdin.read().strip().split()
    if not data:
        return

    n, d, k, max_iter = map(int, data[:4])
    vals = np.array(list(map(float, data[4:])), dtype=float)
    points = vals.reshape(n, d)

    labels, centers = kmeans(points, k, max_iter)
    print(" ".join(map(str, labels.tolist())))
    for row in centers:
        print(" ".join(fmt(x) for x in row))


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
