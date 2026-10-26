import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pickle
import numpy as np
from dataset.mnist import load_mnist
from ch03.sigmoid import sigmoid
from ch03.softmax import softmax


def get_data():
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, y_test


def init_model():
    with open(os.path.join(os.path.dirname(__file__), '../dataset/sample_weight.pkl'), 'rb') as f:
        model = pickle.load(f)
    return model


def predict(model, x):
    W1, W2, W3 = model['W1'], model['W2'], model['W3']
    b1, b2, b3 = model['b1'], model['b2'], model['b3']

    a1 = np.dot(x, W1) + b1  # 1층 선형 변환
    z1 = sigmoid(a1)         # 1층 은닉층 활성화
    a2 = np.dot(z1, W2) + b2 # 2층 선형 변환
    z2 = sigmoid(a2)         # 2층 은닉층 활성화
    a3 = np.dot(z2, W3) + b3 # 3층 선형 변환
    y = softmax(a3)          # 출력을 클래스별 확률로 변환

    return y
