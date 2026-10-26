import numpy as np


# 퍼셉트론으로 AND, NAND, OR, XOR 게이트를 구현한다
# 게이트마다 가중치 w와 편향 b가 다르므로 메서드 안에서 각각 정의한다
class Gate:

    def AND(self, x1, x2):
        w = np.array([0.5, 0.5])
        b = -0.7
        x = np.array([x1, x2])
        total = np.sum(w * x) + b
        return 0 if total <= 0 else 1

    def NAND(self, x1, x2):
        w = np.array([-0.5, -0.5])
        b = 0.7
        x = np.array([x1, x2])
        total = np.sum(w * x) + b
        return 0 if total <= 0 else 1

    def OR(self, x1, x2):
        w = np.array([0.5, 0.5])
        b = -0.2
        x = np.array([x1, x2])
        total = np.sum(w * x) + b
        return 0 if total <= 0 else 1

    # 단층 퍼셉트론으로는 XOR을 만들 수 없으므로 NAND와 OR의 결과를 AND로 합친다 (2층)
    def XOR(self, x1, x2):
        s1 = self.NAND(x1, x2)
        s2 = self.OR(x1, x2)
        return self.AND(s1, s2)


if __name__ == '__main__':
    gate = Gate()
    print(gate.AND(0, 0))  # 0
    print(gate.XOR(0, 1))  # 1
