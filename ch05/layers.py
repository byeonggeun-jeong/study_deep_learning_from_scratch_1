import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from ch03.softmax import softmax
from ch04.loss_functions import cross_entropy_error


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        # x <= 0 인 위치는 출력이 0으로 고정되므로 True로 표시한다
        self.mask = (x <= 0)
        out = x.copy()  # 입력 x를 직접 바꾸지 않도록 복사한다
        out[self.mask] = 0

        return out

    def backward(self, dout):
        # 순전파에서 0으로 고정된 위치는 기울기도 0이다
        dout[self.mask] = 0  # 주의: dout을 제자리에서 수정한다
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out  # backward에서 쓰므로 출력을 저장한다
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out  # 시그모이드 미분 y(1 - y)에 체인 룰 적용
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 가중치와 편향의 기울기
        self.dW = None
        self.db = None

    def forward(self, x):
        # 합성곱 계층의 4차원 입력(N, C, H, W)도 받을 수 있도록 2차원으로 펼친다
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 원래 입력 형상으로 되돌린다
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax 출력
        self.t = None  # 정답 레이블 또는 원-핫 벡터

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 정답이 원-핫 벡터인 경우
            dx = (self.y - self.t) / batch_size
        else:  # 정답이 레이블인 경우
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
