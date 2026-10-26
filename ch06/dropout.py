# [ch06] 드롭아웃 계층
import numpy as np


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None  # 살아남은 뉴런을 기록한다

    def forward(self, x, train_flg=True):
        if train_flg:
            # 난수가 삭제 비율보다 큰 뉴런만 살린다 (True: 살림, False: 삭제)
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            # 시험 때는 뉴런을 삭제하지 않고, 훈련 때 평균적으로 살아남은 비율만큼 출력을 줄인다
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        # 삭제된 뉴런은 기울기도 0이고, 살아남은 뉴런만 같은 mask로 기울기를 통과시킨다
        return dout * self.mask
