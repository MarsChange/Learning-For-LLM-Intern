# 手撕 LayerNorm（NumPy）

## 面试定位

LayerNorm 对每个样本内部的 feature 维度做归一化：

$$
\mu_i=\frac{1}{d}\sum_{j=1}^{d}x_{i,j}
$$

$$
\sigma_i^2=\frac{1}{d}\sum_{j=1}^{d}(x_{i,j}-\mu_i)^2
$$

$$
y_{i,j}=\frac{x_{i,j}-\mu_i}{\sqrt{\sigma_i^2+\epsilon}}
$$

和 BatchNorm 的关键区别：

- BatchNorm：跨 batch 统计，同一个 feature 一组统计量。
- LayerNorm：每个样本内部统计，每个样本一组统计量。

## 输入输出格式

输入：

```text
n d eps affine
X               n 行，每行 d 个数
如果 affine=1:
gamma           d 个数
beta            d 个数
```

输出 `n` 行，每行 `d` 个数。

## 可运行代码

```python
import sys
import numpy as np


def fmt(x: float) -> str:
    if abs(x) < 5e-7:
        x = 0.0
    return f"{x:.6f}"


def layer_norm(x, eps, gamma=None, beta=None):
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=1, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps)
    if gamma is not None and beta is not None:
        y = y * gamma + beta
    return y


def solve():
    data = sys.stdin.read().strip().split()
    if not data:
        return

    n = int(data[0])
    d = int(data[1])
    eps = float(data[2])
    affine = int(data[3])
    idx = 4

    x = np.array(list(map(float, data[idx:idx + n * d])), dtype=float).reshape(n, d)
    idx += n * d

    gamma = beta = None
    if affine:
        gamma = np.array(list(map(float, data[idx:idx + d])), dtype=float)
        idx += d
        beta = np.array(list(map(float, data[idx:idx + d])), dtype=float)

    y = layer_norm(x, eps, gamma, beta)
    for row in y:
        print(" ".join(fmt(v) for v in row))


if __name__ == "__main__":
    solve()
```

## 测试用例

### 用例 1

输入：

```text
2 3 0.00001 0
1 2 3
2 2 2
```

输出：

```text
-1.224736 0.000000 1.224736
0.000000 0.000000 0.000000
```

### 用例 2

输入：

```text
1 2 0.00001 1
1 3
2 3
1 -1
```

输出：

```text
-0.999990 1.999985
```

### 用例 3

输入：

```text
2 2 0.00001 0
1 2
3 5
```

输出：

```text
-0.999980 0.999980
-0.999995 0.999995
```

## 易错点

- LayerNorm 的均值方差沿 feature 维度算。
- `gamma` 和 `beta` 是 feature 维度上的参数，会广播到每个样本。
- 样本内所有 feature 相同，则归一化输出为 0。
