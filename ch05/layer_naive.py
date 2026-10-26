class MulLayer: # 곱셈 계층
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y  # x와 y의 기울기는 서로 바꿔서 곱한다
        dy = dout * self.x

        return dx, dy


class AddLayer: # 덧셈 계층
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        # 덧셈의 미분은 1이므로 상류 기울기를 그대로 전달한다
        dx = dout * 1
        dy = dout * 1

        return dx, dy
