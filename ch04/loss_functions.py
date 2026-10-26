import numpy as np


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 원-핫이면 정답 레이블 인덱스로 변경
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size  # 1e-7: log(0) 방지


if __name__ == '__main__':
    print(mean_squared_error(np.array([0.0, 0.0, 1.0]), np.array([0, 0, 1])))  # 0.0
