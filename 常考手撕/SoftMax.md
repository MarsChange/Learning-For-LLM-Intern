# 手撕 Softmax（NumPy）

## 面试定位

Softmax 把一组 logits 变成概率分布。手撕时重点是数值稳定：

$$
\operatorname{softmax}(x_i)=
\frac{\exp(x_i-\max(x))}
{\sum_j \exp(x_j-\max(x))}
$$

直接 `exp(x)` 容易在大 logits 下溢出或上溢。

## 输入输出格式

输入：

```text
n m axis
n 行，每行 m 个数
```

- `axis = 1`：对每一行做 softmax。
- `axis = 0`：对每一列做 softmax。

输出 `n` 行，每行 `m` 个数，保留 6 位小数。

## 可运行代码

```python
import sys
import numpy as np


def fmt(x: float) -> str:
    if abs(x) < 5e-7:
        x = 0.0
    return f"{x:.6f}"


def softmax(x, axis):
    z = x - np.max(x, axis=axis, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


def solve():
    data = sys.stdin.read().strip().split()
    if not data:
        return

    n, m, axis = map(int, data[:3])
    vals = np.array(list(map(float, data[3:])), dtype=float)
    x = vals.reshape(n, m)
    y = softmax(x, axis)

    for row in y:
        print(" ".join(fmt(v) for v in row))


if __name__ == "__main__":
    solve()
```

## 测试用例

### 用例 1

输入：

```text
2 3 1
1 2 3
1 1 1
```

输出：

```text
0.090031 0.244728 0.665241
0.333333 0.333333 0.333333
```

### 用例 2

输入：

```text
1 3 1
1000 1001 1002
```

输出：

```text
0.090031 0.244728 0.665241
```

### 用例 3

输入：

```text
2 2 0
1 2
3 4
```

输出：

```text
0.119203 0.119203
0.880797 0.880797
```

## 易错点

- 必须减去 `max`，否则大 logits 下 `exp` 可能溢出。
- `axis` 要保留维度，否则广播会错。
- Softmax 输出沿指定维度求和应为 1。
