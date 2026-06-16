# 手撕 Latent Attention 简化版（PyTorch）

## 面试定位

严格的 DeepSeek MLA 细节比面试手撕复杂。手撕题通常考核心思想：

> 不直接缓存完整 K/V，而是先把 hidden state 压缩成低维 latent，再由 latent 恢复 K/V 参与 attention。

这里实现一个“MLA 思想简化版”：

- `latent = X W_down`
- `K = latent W_K_up`
- `V = latent W_V_up`
- Q 仍由 `X W_Q` 得到
- 后续按 MHA 的 scaled dot-product attention 计算

## 输入输出格式

输入：

```text
B T D H R causal
X               共 B*T*D 个数
W_Q             共 D*D 个数
W_down          共 D*R 个数
W_K_up          共 R*D 个数
W_V_up          共 R*D 个数
W_O             共 D*D 个数
```

含义：

- `R` 是 latent 维度。
- `D % H == 0`。

## 可运行代码

```python
import sys
import math
import torch


def fmt(x):
    if abs(x) < 5e-7:
        x = 0.0
    return f"{x:.6f}"


def latent_attention(x, w_q, w_down, w_k_up, w_v_up, w_o, num_heads, causal):
    bsz, seq_len, d_model = x.shape
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    head_dim = d_model // num_heads

    q = (x @ w_q).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    latent = x @ w_down
    k = (latent @ w_k_up).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    v = (latent @ w_v_up).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

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

    b, t, d, h, r, causal = map(int, data[:6])
    vals = list(map(float, data[6:]))
    idx = 0

    def take(cnt):
        nonlocal idx
        part = vals[idx:idx + cnt]
        idx += cnt
        return part

    dtype = torch.float64
    x = torch.tensor(take(b * t * d), dtype=dtype).view(b, t, d)
    wq = torch.tensor(take(d * d), dtype=dtype).view(d, d)
    wd = torch.tensor(take(d * r), dtype=dtype).view(d, r)
    wku = torch.tensor(take(r * d), dtype=dtype).view(r, d)
    wvu = torch.tensor(take(r * d), dtype=dtype).view(r, d)
    wo = torch.tensor(take(d * d), dtype=dtype).view(d, d)

    out = latent_attention(x, wq, wd, wku, wvu, wo, h, bool(causal))
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
1 1 4 2 2 0
1 2 3 4
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
1 0 0 1 0 0 0 0
1 0 0 0 0 1 0 0
1 0 0 0 0 1 0 0
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
```

输出：

```text
1 1 4
1.000000 2.000000 0.000000 0.000000
```

### 用例 2

输入：

```text
1 2 4 2 2 1
1 0 1 0 0 1 0 1
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
1 0 0 1 0 0 0 0
1 0 0 0 0 1 0 0
1 0 0 0 0 1 0 0
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
```

输出：

```text
1 2 4
1.000000 0.000000 0.000000 0.000000
0.330238 0.669762 0.000000 0.000000
```

### 用例 3

输入：

```text
1 2 4 2 2 0
1 0 0 0 0 1 0 0
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
1 0 0 1 0 0 0 0
1 0 0 0 0 1 0 0
1 0 0 0 0 1 0 0
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
```

输出：

```text
1 2 4
0.669762 0.330238 0.000000 0.000000
0.330238 0.669762 0.000000 0.000000
```

## 易错点

- 这是 MLA 思想的手撕简化版，不等同于完整 DeepSeek MLA 工程实现。
- 真正节省 KV Cache 的关键是缓存 latent，而不是缓存完整 K/V。
- `W_K_up` 和 `W_V_up` 要把 latent 恢复到 `D` 维，再拆成 heads。
