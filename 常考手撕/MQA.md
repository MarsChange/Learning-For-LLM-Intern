# 手撕 Multi-Query Attention（PyTorch）

## 面试定位

MQA 和 MHA 的区别是：多个 Q heads 共享同一组 K/V。这样 KV Cache 从 `H` 份降到 1 份。

这里实现的是单层 MQA：

- Q 投影输出维度 `D`
- K/V 投影输出维度 `head_dim = D / H`
- K/V 在 head 维度上只有 1 组，通过广播参与所有 Q heads 的 attention

## 输入输出格式

输入：

```text
B T D H causal
X               共 B*T*D 个数
W_Q             共 D*D 个数
W_K             共 D*head_dim 个数
W_V             共 D*head_dim 个数
W_O             共 D*D 个数
```

输出格式同 MHA。

## 可运行代码

```python
import sys
import math
import torch


def fmt(x):
    if abs(x) < 5e-7:
        x = 0.0
    return f"{x:.6f}"


def mqa(x, w_q, w_k, w_v, w_o, num_q_heads, causal):
    bsz, seq_len, d_model = x.shape
    if d_model % num_q_heads != 0:
        raise ValueError("d_model must be divisible by num_q_heads")

    head_dim = d_model // num_q_heads
    q = (x @ w_q).view(bsz, seq_len, num_q_heads, head_dim).transpose(1, 2)
    k = (x @ w_k).view(bsz, seq_len, 1, head_dim).transpose(1, 2)
    v = (x @ w_v).view(bsz, seq_len, 1, head_dim).transpose(1, 2)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = (attn @ v).transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
    return out @ w_o


def solve():
    data = sys.stdin.read().strip().split()
    if not data:
        return

    b, t, d, h, causal = map(int, data[:5])
    vals = list(map(float, data[5:]))
    head_dim = d // h
    idx = 0

    def take(cnt):
        nonlocal idx
        part = vals[idx:idx + cnt]
        idx += cnt
        return part

    dtype = torch.float64
    x = torch.tensor(take(b * t * d), dtype=dtype).view(b, t, d)
    wq = torch.tensor(take(d * d), dtype=dtype).view(d, d)
    wk = torch.tensor(take(d * head_dim), dtype=dtype).view(d, head_dim)
    wv = torch.tensor(take(d * head_dim), dtype=dtype).view(d, head_dim)
    wo = torch.tensor(take(d * d), dtype=dtype).view(d, d)

    out = mqa(x, wq, wk, wv, wo, h, bool(causal))
    print(b, t, d)
    for row in out.reshape(b * t, d):
        print(" ".join(fmt(v.item()) for v in row))


if __name__ == "__main__":
    solve()
```

## 测试用例

### 用例 1

输入：

```text
1 1 4 2 0
1 2 3 4
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
1 0 0 1 0 0 0 0
1 0 0 1 0 0 0 0
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
```

输出：

```text
1 1 4
1.000000 2.000000 1.000000 2.000000
```

### 用例 2

输入：

```text
1 2 4 2 1
1 0 1 0 0 1 0 1
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
1 0 0 1 0 0 0 0
1 0 0 1 0 0 0 0
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
```

输出：

```text
1 2 4
1.000000 0.000000 1.000000 0.000000
0.330238 0.669762 0.330238 0.669762
```

### 用例 3

输入：

```text
1 2 4 2 0
1 0 0 0 0 1 0 0
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
1 0 0 1 0 0 0 0
1 0 0 1 0 0 0 0
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
```

输出：

```text
1 2 4
0.669762 0.330238 0.500000 0.500000
0.330238 0.669762 0.500000 0.500000
```

## 易错点

- MQA 的 K/V head 数是 1，不是 `H`。
- 输出仍然有 `H` 个 Q heads，最终拼回 `D`。
- PyTorch matmul 会广播 `k/v` 的 head 维度。
