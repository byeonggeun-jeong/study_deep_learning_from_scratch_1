import numpy as np

def softmax(x):
    # 2차원 입력은 샘플마다 (행마다) softmax를 적용한다
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)  # 오버플로 방지: 최댓값을 빼도 결과는 같다
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 오버플로 방지
    return np.exp(x) / np.sum(np.exp(x))
