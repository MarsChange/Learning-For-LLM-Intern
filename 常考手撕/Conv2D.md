# 手撕 Conv2D（NumPy）

## 面试定位

卷积模块常考的是 2D convolution forward，不允许直接调 `torch.nn.Conv2d` 或 `scipy.signal.convolve`。这里实现最常见的 NCHW 输入和 OIHW 权重：

$$
Y_{n,f,i,j}=
\sum_{c=0}^{C-1}
\sum_{u=0}^{K_H-1}
\sum_{v=0}^{K_W-1}
X^{pad}_{n,c,i\cdot s+u,j\cdot s+v}
W_{f,c,u,v}
+ b_f
$$

输出尺寸：

$$
H_{out}=\left\lfloor\frac{H+2p-K_H}{s}\right\rfloor+1
$$

$$
W_{out}=\left\lfloor\frac{W+2p-K_W}{s}\right\rfloor+1
$$

这里实现的是深度学习里的 cross-correlation 形式，也就是卷积核不翻转；这和 PyTorch `Conv2d` 的 forward 行为一致。

## 输入输出格式

输入：

```text
N C H W
F KH KW stride pad use_bias
X       共 N*C*H*W 个数，按 NCHW 顺序
Weight  共 F*C*KH*KW 个数，按 OIHW 顺序
如果 use_bias=1:
bias    共 F 个数
```

输出：

```text
N F H_out W_out
按 n -> f -> h 的顺序输出，每行 W_out 个数
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


def conv2d(x, weight, bias=None, stride=1, pad=0):
    # x: [N, C, H, W]
    # weight: [F, C, KH, KW]，F 是输出通道数，也就是卷积核个数。
    n, c, h, w = x.shape
    f, wc, kh, kw = weight.shape
    assert c == wc

    out_h = (h + 2 * pad - kh) // stride + 1
    out_w = (w + 2 * pad - kw) // stride + 1

    # 只在 H/W 两个空间维度补零，batch 和 channel 维度不补。
    x_pad = np.pad(
        x,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0.0,
    )

    out = np.zeros((n, f, out_h, out_w), dtype=float)

    for ni in range(n):
        for fi in range(f):
            for i in range(out_h):
                hs = i * stride
                for j in range(out_w):
                    ws = j * stride
                    # window: [C, KH, KW]，和 weight[fi] 逐元素相乘后求和。
                    window = x_pad[ni, :, hs:hs + kh, ws:ws + kw]
                    out[ni, fi, i, j] = np.sum(window * weight[fi])
                    if bias is not None:
                        out[ni, fi, i, j] += bias[fi]

    return out


def solve():
    # ACM 模式：一次性读入所有 token，然后按输入格式切片。
    data = sys.stdin.read().strip().split()
    if not data:
        return

    n, c, h, w = map(int, data[:4])
    f, kh, kw, stride, pad, use_bias = map(int, data[4:10])
    vals = list(map(float, data[10:]))
    idx = 0

    def take(cnt):
        # 从扁平数组中取出 cnt 个数，并移动读取指针。
        nonlocal idx
        part = vals[idx:idx + cnt]
        idx += cnt
        return part

    # 恢复 NCHW 输入和 OIHW 卷积核。
    x = np.array(take(n * c * h * w), dtype=float).reshape(n, c, h, w)
    weight = np.array(take(f * c * kh * kw), dtype=float).reshape(f, c, kh, kw)

    bias = None
    if use_bias:
        bias = np.array(take(f), dtype=float)

    out = conv2d(x, weight, bias=bias, stride=stride, pad=pad)

    print(out.shape[0], out.shape[1], out.shape[2], out.shape[3])
    for ni in range(out.shape[0]):
        for fi in range(out.shape[1]):
            for i in range(out.shape[2]):
                print(" ".join(fmt(v) for v in out[ni, fi, i]))


if __name__ == "__main__":
    solve()
```

## 测试用例

### 用例 1

单通道、单卷积核、无 padding、无 bias。

输入：

```text
1 1 3 3
1 2 2 1 0 0
1 2 3 4 5 6 7 8 9
1 0 0 -1
```

输出：

```text
1 1 2 2
-4.000000 -4.000000
-4.000000 -4.000000
```

### 用例 2

多输入通道，带 bias。

输入：

```text
1 2 2 2
1 2 2 1 0 1
1 2 3 4 10 20 30 40
1 1 1 1 0.1 0.1 0.1 0.1
1
```

输出：

```text
1 1 1 1
21.000000
```

### 用例 3

stride=2，pad=1。

输入：

```text
1 1 3 3
1 3 3 2 1 0
1 2 3 4 5 6 7 8 9
1 1 1 1 1 1 1 1 1
```

输出：

```text
1 1 2 2
12.000000 16.000000
24.000000 28.000000
```

## 易错点

- 深度学习框架里的 `Conv2d` 通常做 cross-correlation，卷积核不翻转。
- 输入是 NCHW，权重是 OIHW：`[输出通道数, 输入通道数, 卷积核高, 卷积核宽]`。
- padding 只补空间维度 H/W，不补 batch/channel。
- `out_h` 和 `out_w` 都要向下取整，且要求窗口不能越过 padding 后的边界。
- 多通道卷积不是每个通道各自输出一张图，而是先对每个输入通道做窗口乘加，再在通道维度求和，得到一个输出通道。
