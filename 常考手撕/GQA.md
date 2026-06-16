# 手撕 Grouped-Query Attention（PyTorch）

## 面试定位

GQA 是 MHA 和 MQA 的折中：

- Q 有 `H_q` 个 heads。
- K/V 有 `H_kv` 个 heads。
- 要求 `H_q % H_kv == 0`。
- 每 `H_q / H_kv` 个 Q heads 共享同一个 K/V head。

实现时最直接的写法是把 K/V 用 `repeat_interleave` 扩展到 `H_q` 个 heads，再按 MHA 计算。

## 输入输出格式

输入：

```text
B T D H_q H_kv causal
X               共 B*T*D 个数
W_Q             共 D*D 个数
W_K             共 D*(H_kv*head_dim) 个数
W_V             共 D*(H_kv*head_dim) 个数
W_O             共 D*D 个数
```

其中 `head_dim = D / H_q`。

## 可运行代码

```python
import sys
import math
import torch


def fmt(x):
    if abs(x) < 5e-7:
        x = 0.0
    return f"{x:.6f}"


def gqa(x, w_q, w_k, w_v, w_o, num_q_heads, num_kv_heads, causal):
    bsz, seq_len, d_model = x.shape
    if d_model % num_q_heads != 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError("invalid head numbers")

    head_dim = d_model // num_q_heads
    repeat = num_q_heads // num_kv_heads

    q = (x @ w_q).view(bsz, seq_len, num_q_heads, head_dim).transpose(1, 2)
    k = (x @ w_k).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = (x @ w_v).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    out = (torch.softmax(scores, dim=-1) @ v)
    out = out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
    return out @ w_o


def solve():
    data = sys.stdin.read().strip().split()
    if not data:
        return

    b, t, d, hq, hkv, causal = map(int, data[:6])
    vals = list(map(float, data[6:]))
    head_dim = d // hq
    idx = 0

    def take(cnt):
        nonlocal idx
        part = vals[idx:idx + cnt]
        idx += cnt
        return part

    dtype = torch.float64
    x = torch.tensor(take(b * t * d), dtype=dtype).view(b, t, d)
    wq = torch.tensor(take(d * d), dtype=dtype).view(d, d)
    wk = torch.tensor(take(d * hkv * head_dim), dtype=dtype).view(d, hkv * head_dim)
    wv = torch.tensor(take(d * hkv * head_dim), dtype=dtype).view(d, hkv * head_dim)
    wo = torch.tensor(take(d * d), dtype=dtype).view(d, d)

    out = gqa(x, wq, wk, wv, wo, hq, hkv, bool(causal))
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
1 1 4 4 2 0
1 2 3 4
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
1 0 0 1 0 0 0 0
1 0 0 1 0 0 0 0
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
```

输出：

```text
1 1 4
1.000000 1.000000 2.000000 2.000000
```

### 用例 2

输入：

```text
1 2 4 4 2 1
1 0 1 0 0 1 0 1
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
1 0 0 1 0 0 0 0
1 0 0 1 0 0 0 0
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
```

输出：

```text
1 2 4
1.000000 1.000000 0.000000 0.000000
0.500000 0.731059 0.500000 0.731059
```

### 用例 3

输入：

```text
1 2 4 4 2 0
1 0 0 0 0 1 0 0
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
1 0 0 1 0 0 0 0
1 0 0 1 0 0 0 0
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
```

输出：

```text
1 2 4
0.731059 0.500000 0.500000 0.500000
0.500000 0.731059 0.500000 0.500000
```

## 易错点

- `head_dim` 由 Q heads 决定：`D / H_q`。
- K/V 投影维度是 `H_kv * head_dim`。
- `repeat_interleave` 是沿 head 维复制，不是沿 sequence 维复制。
