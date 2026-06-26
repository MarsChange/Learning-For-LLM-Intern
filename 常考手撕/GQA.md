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
    # 固定输出格式，避免浮点误差导致 -0.000000。
    if abs(x) < 5e-7:
        x = 0.0
    return f"{x:.6f}"


def gqa(x, w_q, w_k, w_v, w_o, num_q_heads, num_kv_heads, causal):
    # x: [B, T, D]。GQA 中 Q head 数通常多于 K/V head 数。
    bsz, seq_len, d_model = x.shape
    if d_model % num_q_heads != 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError("invalid head numbers")

    # head_dim 由 Q heads 决定；K/V 每个 head 也使用同样的 head_dim。
    head_dim = d_model // num_q_heads
    # 每个 K/V head 要服务多少个 Q heads。
    repeat = num_q_heads // num_kv_heads

    # Q: [B, T, Hq*head_dim] = [B, T, D]，再拆成 [B, Hq, T, head_dim]。
    q = (x @ w_q).view(bsz, seq_len, num_q_heads, head_dim).transpose(1, 2)
    # K/V 只投影到 Hkv 个 head，总维度是 Hkv*head_dim，小于 D。
    k = (x @ w_k).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = (x @ w_v).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # 把 K/V 沿 head 维复制到 Hq 个 head：
    # [B, Hkv, T, head_dim] -> [B, Hq, T, head_dim]
    # 这样后续就能直接复用 MHA 的 attention 计算。
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)

    # 每个 Q head 与对应复制出来的 K/V head 做 scaled dot-product attention。
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    if causal:
        # mask 会从 [T, T] 广播到 [B, Hq, T, T]。
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    # softmax 后乘 V，得到 [B, Hq, T, head_dim]。
    out = (torch.softmax(scores, dim=-1) @ v)
    # 拼回 [B, T, D]，再做输出投影。
    out = out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
    return out @ w_o


def solve():
    # 按 ACM 输入格式解析：先读超参数，再按顺序读矩阵。
    data = sys.stdin.read().strip().split()
    if not data:
        return

    b, t, d, hq, hkv, causal = map(int, data[:6])
    vals = list(map(float, data[6:]))
    head_dim = d // hq
    idx = 0

    def take(cnt):
        # 取出一段扁平参数，并推进 idx。
        nonlocal idx
        part = vals[idx:idx + cnt]
        idx += cnt
        return part

    dtype = torch.float64
    x = torch.tensor(take(b * t * d), dtype=dtype).view(b, t, d)
    wq = torch.tensor(take(d * d), dtype=dtype).view(d, d)
    # GQA 的 K/V 投影矩阵不是 D*D，而是 D*(Hkv*head_dim)。
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
