import numpy as np


class BatchNormalization:

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma  # 학습되는 스케일
        self.beta = beta    # 학습되는 시프트
        self.momentum = momentum
        self.input_shape = None  # 합성곱 계층은 4차원, 완전연결 계층은 2차원

        # 시험 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var

        # backward에서 쓰는 중간 데이터
        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward_preprocess(self, x, train_flg=True):
        self.input_shape = x.shape
        # 4차원 입력(N, C, H, W)은 (N, C*H*W) 2차원으로 펴서 처리한다
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def forward(self, x, train_flg):
        # 시험 통계가 아직 없으면(첫 호출) 0으로 초기화한다
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu                  # 평균 중심화
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 10e-7)   # 0 나눗셈 방지
            xn = xc / std                # 정규화 x̂

            # backward에서 쓸 중간 값을 저장한다
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            # 시험 때 쓸 평균과 분산을 이동 평균으로 갱신한다
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 10e-7))

        out = self.gamma * xn + self.beta  # y = γ·x̂ + β
        return out

    def backward_preprocess(self, dout):
        # forward와 같이 2차원으로 펴서 역전파하고, 원래 입력 모양으로 되돌려 반환한다
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.backward(dout)
        return dx.reshape(*self.input_shape)

    def backward(self, dout):
        # β의 기울기는 열 방향 합, γ의 기울기는 x̂와의 곱의 합이다
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        # x̂ → xc → std/var → μ → x 순서로 기울기를 전파한다
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar  # 분산을 거쳐 xc로 들어오는 기울기
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size # 평균 μ를 거쳐 들어오는 기울기

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx
