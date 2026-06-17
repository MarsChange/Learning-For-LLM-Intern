# 手撕 Self-Attention（NumPy）

## 面试定位

单头 Self-Attention 的核心：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(
\frac{QK^T}{\sqrt{d_k}} + M
\right)V
$$

这里不用 PyTorch，高级封装也不调用，只用 NumPy 手写。

## 输入输出格式

输入：

```text
T D causal
X       共 T*D 个数
W_Q     共 D*D 个数
W_K     共 D*D 个数
W_V     共 D*D 个数
```

- `T`：序列长度。
- `D`：hidden size。
- `causal = 1`：使用 causal mask。
- `causal = 0`：不使用 mask。

输出 `T` 行，每行 `D` 个数。

## 可运行代码

```python
import sys
import math
import numpy as np


def fmt(x: float) -> str:
    if abs(x) < 5e-7:
        x = 0.0
    return f"{x:.6f}"


def softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


def self_attention(x, w_q, w_k, w_v, causal):
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    scores = (q @ k.T) / math.sqrt(q.shape[-1])
    if causal:
        mask = np.triu(np.ones_like(scores, dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)

    attn = softmax(scores, axis=1)
    return attn @ v


def solve():
    data = sys.stdin.read().strip().split()
    if not data:
        return

    t, d, causal = map(int, data[:3])
    vals = list(map(float, data[3:]))
    idx = 0

    def take(cnt):
        nonlocal idx
        part = vals[idx:idx + cnt]
        idx += cnt
        return part

    x = np.array(take(t * d), dtype=float).reshape(t, d)
    w_q = np.array(take(d * d), dtype=float).reshape(d, d)
    w_k = np.array(take(d * d), dtype=float).reshape(d, d)
    w_v = np.array(take(d * d), dtype=float).reshape(d, d)

    out = self_attention(x, w_q, w_k, w_v, bool(causal))
    for row in out:
        print(" ".join(fmt(v) for v in row))


if __name__ == "__main__":
    solve()
```

## 测试用例

### 用例 1

输入：

```text
2 2 1
1 0
0 1
1 0 0 1
1 0 0 1
1 0 0 1
```

输出：

```text
1.000000 0.000000
0.330238 0.669762
```

### 用例 2

输入：

```text
2 2 0
1 0
0 1
1 0 0 1
1 0 0 1
1 0 0 1
```

输出：

```text
0.669762 0.330238
0.330238 0.669762
```

### 用例 3

输入：

```text
3 2 1
1 0
1 0
0 1
1 0 0 1
1 0 0 1
1 0 0 1
```

输出：

```text
1.000000 0.000000
1.000000 0.000000
0.496510 0.503490
```

## 易错点

- 缩放因子是 `sqrt(d_k)`。
- causal mask 是上三角，不包含对角线。
- Softmax 在 key 维度上做，也就是每一行归一化。
