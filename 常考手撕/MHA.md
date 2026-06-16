# 手撕 Multi-Head Attention（PyTorch）

## 面试定位

MHA 常考的是 shape 变换和 mask：

- `Q = XW_Q, K = XW_K, V = XW_V`
- 拆成 `num_heads` 个 head
- 计算 `softmax(QK^T / sqrt(head_dim))V`
- 拼回 `d_model`
- 过输出投影 `W_O`

这里给 ACM 模式代码：从标准输入读矩阵，输出 attention 结果。实现允许使用 PyTorch。

## 输入输出格式

输入：

```text
B T D H causal
X               共 B*T*D 个数
W_Q             共 D*D 个数
W_K             共 D*D 个数
W_V             共 D*D 个数
W_O             共 D*D 个数
```

含义：

- `B`：batch size
- `T`：sequence length
- `D`：hidden size
- `H`：attention heads，要求 `D % H == 0`
- `causal`：`1` 表示使用 causal mask，`0` 表示不使用

输出：

```text
B T D
第 1 个 token 的 D 维输出
...
第 B*T 个 token 的 D 维输出
```

保留 6 位小数。

## 可运行代码

```python
import sys
import math
import torch


def fmt(x: float) -> str:
    if abs(x) < 5e-7:
        x = 0.0
    return f"{x:.6f}"


def multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads, causal):
    bsz, seq_len, d_model = x.shape
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    head_dim = d_model // num_heads

    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = attn @ v
    out = out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
    return out @ w_o


def take(vals, idx, cnt):
    return vals[idx:idx + cnt], idx + cnt


def solve():
    data = sys.stdin.read().strip().split()
    if not data:
        return

    bsz, seq_len, d_model, num_heads, causal = map(int, data[:5])
    vals = list(map(float, data[5:]))
    idx = 0

    x_vals, idx = take(vals, idx, bsz * seq_len * d_model)
    wq_vals, idx = take(vals, idx, d_model * d_model)
    wk_vals, idx = take(vals, idx, d_model * d_model)
    wv_vals, idx = take(vals, idx, d_model * d_model)
    wo_vals, idx = take(vals, idx, d_model * d_model)

    dtype = torch.float64
    x = torch.tensor(x_vals, dtype=dtype).view(bsz, seq_len, d_model)
    w_q = torch.tensor(wq_vals, dtype=dtype).view(d_model, d_model)
    w_k = torch.tensor(wk_vals, dtype=dtype).view(d_model, d_model)
    w_v = torch.tensor(wv_vals, dtype=dtype).view(d_model, d_model)
    w_o = torch.tensor(wo_vals, dtype=dtype).view(d_model, d_model)

    out = multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads, bool(causal))
    print(bsz, seq_len, d_model)
    for row in out.reshape(bsz * seq_len, d_model):
        print(" ".join(fmt(v.item()) for v in row))


if __name__ == "__main__":
    solve()
```

## 测试用例

### 用例 1

输入：

```text
1 1 2 1 0
1 2
1 0 0 1
1 0 0 1
1 0 0 1
1 0 0 1
```

输出：

```text
1 1 2
1.000000 2.000000
```

### 用例 2

输入：

```text
1 2 2 1 1
1 0 0 1
1 0 0 1
1 0 0 1
1 0 0 1
1 0 0 1
```

输出：

```text
1 2 2
1.000000 0.000000
0.330238 0.669762
```

### 用例 3

输入：

```text
1 2 4 2 0
1 0 1 0 0 1 0 1
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1
```

输出：

```text
1 2 4
0.669762 0.330238 0.669762 0.330238
0.330238 0.669762 0.330238 0.669762
```

## 易错点

- `view` 前后要理解维度：`[B,T,D] -> [B,H,T,head_dim]`。
- `transpose` 后最好接 `contiguous()` 再 `view`。
- causal mask 要 mask 未来位置，也就是上三角不含对角线。
- softmax 维度是最后一维，即每个 query 对所有 key 的分布。
