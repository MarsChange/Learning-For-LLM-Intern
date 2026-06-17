# 手撕交叉熵损失（CrossEntropy, NumPy）

## 面试定位

分类交叉熵通常等价于：

$$
\operatorname{CrossEntropy}(\text{logits}, y)
=-\log \operatorname{softmax}(\text{logits})_y
$$

手撕时不要先显式算 softmax 再取 log，推荐使用稳定的 log-sum-exp：

$$
z_i = \text{logits}_i-\max_j(\text{logits}_j)
$$

$$
\log \operatorname{softmax}_i
= z_i-\log\sum_j \exp(z_j)
$$

## 输入输出格式

输入：

```text
n c reduction
n 行 logits，每行 c 个数
1 行 labels，共 n 个整数
```

- `reduction = 0`：输出 mean loss。
- `reduction = 1`：输出 sum loss。
- `reduction = 2`：输出每个样本的 loss。

输出保留 6 位小数。

## 可运行代码

```python
import sys
import numpy as np


def fmt(x: float) -> str:
    if abs(x) < 5e-7:
        x = 0.0
    return f"{x:.6f}"


def cross_entropy(logits, labels):
    z = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(z), axis=1, keepdims=True))
    log_probs = z - logsumexp
    return -log_probs[np.arange(len(labels)), labels]


def solve():
    data = sys.stdin.read().strip().split()
    if not data:
        return

    n, c, reduction = map(int, data[:3])
    idx = 3
    logits = np.array(list(map(float, data[idx:idx + n * c])), dtype=float).reshape(n, c)
    idx += n * c
    labels = np.array(list(map(int, data[idx:idx + n])), dtype=int)

    losses = cross_entropy(logits, labels)
    if reduction == 0:
        print(fmt(float(np.mean(losses))))
    elif reduction == 1:
        print(fmt(float(np.sum(losses))))
    else:
        print(" ".join(fmt(v) for v in losses))


if __name__ == "__main__":
    solve()
```

## 测试用例

### 用例 1

输入：

```text
2 3 0
1 2 3
1 1 1
2 0
```

输出：

```text
0.753109
```

### 用例 2

输入：

```text
1 3 2
1000 1001 1002
0
```

输出：

```text
2.407606
```

### 用例 3

输入：

```text
3 2 1
2 0
0 2
1 1
0 1 1
```

输出：

```text
0.947003
```

## 易错点

- 标签是类别下标，不是 one-hot。
- CrossEntropy 输入通常是 logits，不需要先 softmax。
- 大 logits 下要用 log-sum-exp 稳定计算。
