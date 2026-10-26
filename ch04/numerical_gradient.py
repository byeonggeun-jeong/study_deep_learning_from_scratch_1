import numpy as np


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)  # 중심 차분


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    # f 안에서 참조하는 값(예: 신경망 가중치)에 반영되도록 원본 x를 직접 바꾼다
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        val = x[idx]
        x[idx] = val + h
        fxh1 = f(x)

        x[idx] = val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = val  # 원래 값 복원
        it.iternext()

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x.astype(float)  # 복사본을 써서 호출자의 배열이 바뀌지 않게

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


if __name__ == '__main__':
    print(gradient_descent(lambda x: np.sum(x ** 2), np.array([-3.0, 4.0]), lr=0.1, step_num=100))