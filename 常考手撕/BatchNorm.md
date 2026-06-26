# 手撕 BatchNorm（NumPy）

## 面试定位

BatchNorm 对一个 batch 的同一 feature 维度做归一化：

$$
\mu_j=\frac{1}{n}\sum_{i=1}^{n}x_{i,j}
$$

$$
\sigma_j^2=\frac{1}{n}\sum_{i=1}^{n}(x_{i,j}-\mu_j)^2
$$

$$
y_{i,j}=\frac{x_{i,j}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}}
$$

如果有仿射参数：

$$
\operatorname{out}_{i,j}=\gamma_j y_{i,j}+\beta_j
$$

这里实现训练阶段的 BatchNorm，不包含 running mean/var。

## 输入输出格式

输入：

```text
n d eps affine
X               n 行，每行 d 个数
如果 affine=1:
gamma           d 个数
beta            d 个数
```

输出 `n` 行，每行 `d` 个数，保留 6 位小数。

## 可运行代码

```python
import sys
import numpy as np


def fmt(x: float) -> str:
    # 输出格式对齐判题要求；把极小浮点误差视为 0。
    if abs(x) < 5e-7:
        x = 0.0
    return f"{x:.6f}"


def batch_norm(x, eps, gamma=None, beta=None):
    # x: [n, d]。BatchNorm 对 batch 维度做统计：
    # 每个 feature j 用 n 个样本上的均值/方差。
    mean = np.mean(x, axis=0, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=0, keepdims=True)
    # 标准化时 mean/var 形状是 [1,d]，会广播到 [n,d]。
    y = (x - mean) / np.sqrt(var + eps)
    if gamma is not None and beta is not None:
        # gamma/beta 是每个 feature 一套参数，shape [d]。
        y = y * gamma + beta
    return y


def solve():
    # 解析输入：n 个样本，每个样本 d 个 feature。
    data = sys.stdin.read().strip().split()
    if not data:
        return

    n = int(data[0])
    d = int(data[1])
    eps = float(data[2])
    affine = int(data[3])
    idx = 4

    # 读取输入矩阵 X。
    x = np.array(list(map(float, data[idx:idx + n * d])), dtype=float).reshape(n, d)
    idx += n * d

    gamma = beta = None
    if affine:
        # 如果使用仿射变换，则继续读取 gamma 和 beta。
        gamma = np.array(list(map(float, data[idx:idx + d])), dtype=float)
        idx += d
        beta = np.array(list(map(float, data[idx:idx + d])), dtype=float)

    # 这里实现的是训练阶段 BatchNorm：直接用当前 batch 统计量。
    y = batch_norm(x, eps, gamma, beta)
    for row in y:
        print(" ".join(fmt(v) for v in row))


if __name__ == "__main__":
    solve()
```

## 测试用例

### 用例 1

输入：

```text
2 2 0.00001 0
1 2
3 4
```

输出：

```text
-0.999995 -0.999995
0.999995 0.999995
```

### 用例 2

输入：

```text
2 2 0.00001 1
1 2
3 4
2 3
1 -1
```

输出：

```text
-0.999990 -3.999985
2.999990 1.999985
```

### 用例 3

输入：

```text
2 2 0.00001 0
1 1
1 1
```

输出：

```text
0.000000 0.000000
0.000000 0.000000
```

## 易错点

- BatchNorm 是跨 batch 维度统计，每个 feature 单独算均值方差。
- 这里用的是总体方差，即除以 `n`，不是无偏方差。
- `eps` 放在方差里面：`sqrt(var + eps)`。
