import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from dataset.mnist import load_mnist
from ch05.two_layer_net import TwoLayerNet

(x_train, t_train), _ = load_mnist(normalize=True, one_hot_label=True)

data = x_train[:3]
label = t_train[:3]

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

grad_numerical = network.numerical_gradient(data, label)
grad_backprop = network.gradient(data, label)

# 두 기울기의 평균 절대 오차가 아주 작으면 역전파 구현 검증 OK
for i in ('W1', 'b1', 'W2', 'b2'):
    diff = np.average(np.abs(grad_backprop[i] - grad_numerical[i]))
    print(i + ':' + str(diff))
