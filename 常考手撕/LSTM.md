# 手撕 LSTM（NumPy）

## 面试定位

LSTM 的核心是 4 个门：

$$
i_t=\sigma(x_tW_{ii}+h_{t-1}W_{hi}+b_i)
$$

$$
f_t=\sigma(x_tW_{if}+h_{t-1}W_{hf}+b_f)
$$

$$
g_t=\tanh(x_tW_{ig}+h_{t-1}W_{hg}+b_g)
$$

$$
o_t=\sigma(x_tW_{io}+h_{t-1}W_{ho}+b_o)
$$

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t
$$

$$
h_t=o_t\odot\tanh(c_t)
$$

这里把 4 个门的权重拼在一起，门顺序固定为：

```text
i, f, g, o
```

## 输入输出格式

输入：

```text
T I H
X        共 T*I 个数
W_ih     共 I*(4H) 个数
W_hh     共 H*(4H) 个数
b        共 4H 个数
h0       共 H 个数
c0       共 H 个数
```

输出：

```text
T 行 hidden state h_t
1 行 final cell state c_T
```

所有数保留 6 位小数。

## 可运行代码

```python
import sys
import numpy as np


def fmt(x: float) -> str:
    # 消除 -0.000000，保持输出格式稳定。
    if abs(x) < 5e-7:
        x = 0.0
    return f"{x:.6f}"


def sigmoid(x):
    # LSTM 的输入门、遗忘门、输出门都使用 sigmoid，输出范围 (0,1)。
    return 1.0 / (1.0 + np.exp(-x))


def lstm(x, w_ih, w_hh, b, h, c):
    # x: [T, I]，w_ih: [I, 4H]，w_hh: [H, 4H]
    # h/c 分别是上一时刻 hidden state 和 cell state，shape 都是 [H]。
    hidden_size = h.shape[0]
    outputs = []

    for xt in x:
        # 一次性算出 4 个门的 pre-activation，shape [4H]。
        # 门顺序固定为 i, f, g, o。
        gates = xt @ w_ih + h @ w_hh + b
        # input gate：控制候选记忆 g 有多少写入 cell。
        i = sigmoid(gates[:hidden_size])
        # forget gate：控制旧 cell state 保留多少。
        f = sigmoid(gates[hidden_size:2 * hidden_size])
        # candidate gate：候选记忆内容，范围 (-1,1)。
        g = np.tanh(gates[2 * hidden_size:3 * hidden_size])
        # output gate：控制 cell state 暴露到 hidden state 的比例。
        o = sigmoid(gates[3 * hidden_size:])

        # cell state 是长期记忆：旧记忆经过 f 保留，新候选经过 i 写入。
        c = f * c + i * g
        # hidden state 是当前输出：先 tanh 压缩 cell，再乘输出门。
        h = o * np.tanh(c)
        # h 会在下一时刻继续使用；copy 避免后续更新影响已保存输出。
        outputs.append(h.copy())

    return np.array(outputs), c


def solve():
    # 读取序列长度 T、输入维度 I、隐藏维度 H。
    data = sys.stdin.read().strip().split()
    if not data:
        return

    t = int(data[0])
    input_size = int(data[1])
    hidden_size = int(data[2])
    vals = list(map(float, data[3:]))
    idx = 0

    def take(cnt):
        # 顺序取 cnt 个数，简化扁平输入的矩阵恢复。
        nonlocal idx
        part = vals[idx:idx + cnt]
        idx += cnt
        return part

    # 按题目约定恢复 X、输入权重、循环权重、bias、初始 h/c。
    x = np.array(take(t * input_size), dtype=float).reshape(t, input_size)
    w_ih = np.array(take(input_size * 4 * hidden_size), dtype=float).reshape(input_size, 4 * hidden_size)
    w_hh = np.array(take(hidden_size * 4 * hidden_size), dtype=float).reshape(hidden_size, 4 * hidden_size)
    b = np.array(take(4 * hidden_size), dtype=float)
    h0 = np.array(take(hidden_size), dtype=float)
    c0 = np.array(take(hidden_size), dtype=float)

    # outputs: 每个时刻的 h_t；c 是最后一个时刻的 cell state。
    outputs, c = lstm(x, w_ih, w_hh, b, h0, c0)
    for row in outputs:
        print(" ".join(fmt(v) for v in row))
    print(" ".join(fmt(v) for v in c))


if __name__ == "__main__":
    solve()
```

## 测试用例

### 用例 1

输入：

```text
1 1 1
1
1 0 1 1
0 0 0 0
0 0 0 0
0
0
```

输出：

```text
0.369606
0.556770
```

### 用例 2

输入：

```text
2 1 1
1
2
1 1 1 1
0 0 0 0
0 0 0 0
0
0
```

输出：

```text
0.369606
0.767664
1.339514
```

### 用例 3

输入：

```text
2 2 1
1 0
0 1
1 0 1 0
0 1 0 1
0 0 0 0
0 0 0 0
0
0
```

输出：

```text
0.252788
0.282151
0.407031
```

## 易错点

- 门顺序要固定。这里是 `i, f, g, o`，和部分框架实现可能不同。
- `c_t` 是长期记忆，`h_t` 是输出 hidden state。
- `g_t` 用 `tanh`，其他三个门用 `sigmoid`。
