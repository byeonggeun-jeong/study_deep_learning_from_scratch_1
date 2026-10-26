import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from collections import OrderedDict
from ch05.layers import Affine, Relu, SoftmaxWithLoss
from ch04.numerical_gradient import numerical_gradient


class TwoLayerNet:
    """2층 신경망
    - 입력층 → 은닉층 → 출력층
    - 활성화 함수: ReLU, 출력층: Softmax
    - 손실 함수: 교차 엔트로피 오차
    """
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)  # (784, 50)
        self.params['b1'] = np.zeros(hidden_size)                                         # (50,)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)  # (50, 10)
        self.params['b2'] = np.zeros(output_size)                                         # (10,)

        # 순서가 중요하므로 OrderedDict에 계층을 순서대로 넣는다
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 수치 미분으로 구한 기울기 (역전파 결과와 비교용)
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    # 역전파로 기울기를 구한다: 순전파로 중간값을 저장한 뒤 계층을 역순으로 통과시킨다
    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
