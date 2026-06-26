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
    # 判题输出通常要求固定小数位；极小数直接当 0，避免打印 -0.000000。
    if abs(x) < 5e-7:
        x = 0.0
    return f"{x:.6f}"


def softmax(x, axis=-1):
    # 数值稳定 softmax：先减去当前维度最大值，避免 exp(大数) 溢出。
    z = x - np.max(x, axis=axis, keepdims=True)
    exp_z = np.exp(z)
    # keepdims=True 保持维度，方便和原矩阵按行/列广播相除。
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


def self_attention(x, w_q, w_k, w_v, causal):
    # x: [T, D]，T 是序列长度，D 是 hidden size。
    # 三个线性投影仍保持 [T, D]：每个 token 都得到自己的 Q/K/V。
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    # scores: [T, T]，第 i 行第 j 列表示 token i 对 token j 的注意力打分。
    # 除以 sqrt(d_k) 是为了稳定 softmax 的输入尺度。
    scores = (q @ k.T) / math.sqrt(q.shape[-1])
    if causal:
        # causal mask 屏蔽未来 token：上三角 k=1 表示不包含主对角线。
        # mask=True 的位置会被置为 -inf，softmax 后概率变成 0。
        mask = np.triu(np.ones_like(scores, dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)

    # 对每个 query 的所有 key 做归一化，所以 axis=1，即每一行和为 1。
    attn = softmax(scores, axis=1)
    # [T, T] @ [T, D] -> [T, D]：用注意力权重加权求和 V。
    return attn @ v


def solve():
    # ACM 模式：一次性读入所有 token，按约定顺序手动切片。
    data = sys.stdin.read().strip().split()
    if not data:
        return

    t, d, causal = map(int, data[:3])
    vals = list(map(float, data[3:]))
    idx = 0

    def take(cnt):
        # 从扁平数组中取出 cnt 个数，并把读取指针向后移动。
        nonlocal idx
        part = vals[idx:idx + cnt]
        idx += cnt
        return part

    # 依次恢复输入矩阵和三个投影矩阵。
    x = np.array(take(t * d), dtype=float).reshape(t, d)
    w_q = np.array(take(d * d), dtype=float).reshape(d, d)
    w_k = np.array(take(d * d), dtype=float).reshape(d, d)
    w_v = np.array(take(d * d), dtype=float).reshape(d, d)

    # bool(causal): 输入 1/0 转成 True/False。
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
