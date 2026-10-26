import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from ch03.sigmoid import sigmoid
from ch03.softmax import softmax
from ch04.loss_functions import cross_entropy_error
from ch04.numerical_gradient import numerical_gradient


class TwoLayerNet:
    # 구조: 입력 x -> [Affine(W1, b1) -> Sigmoid] -> [Affine(W2, b2) -> Softmax] -> 출력 y
    # 손실은 교차 엔트로피, 기울기는 수치 미분으로 구한다
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1  # 1층 선형 변환
        z1 = sigmoid(a1)         # 은닉층 활성화
        a2 = np.dot(z1, W2) + b2 # 2층 선형 변환
        y = softmax(a2)          # 클래스별 확률로 변환

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)  # 파라미터를 하나씩 바꿔가며 loss를 계산하는 함수

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
