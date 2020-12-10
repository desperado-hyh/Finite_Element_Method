import sympy
import matplotlib.pyplot as plt
import numpy as np

xjm1, xj, xjp1 = sympy.symbols(['x_{j-1}', 'x_{j}', 'x_{j+1}'])
x, h, eps = sympy.symbols(['x', 'h', 'epsilon'])
rho = lambda x: sympy.sin(sympy.pi * x)

A = - sympy.integrate(rho(x) * (x  - xjm1)/h, (x, xjm1, xj))
B = - sympy.integrate(rho(x) * (xjp1 - x)/h, (x, xj, xjp1))

print(sympy.simplify(A + B))

from scipy.sparse import dia_matrix
from scipy.sparse.linalg import inv
from numpy import pi

class FEM:
    def __init__(self, nodes, xmin=0, xmax=1):
        self.nodes = nodes
        x = np.linspace(xmin, max, nodes)
        self.x = x
        self.h = x[1] - x[0]

    def Kmatrix(self):
        n = self.nodes
        m = 1/self.h * np.ones(n)
        data = [m, -2*m, m]
        offsets = [-1, 0, 1]
        # 使用scipy.sparse稀疏矩阵库，构造3对角稀疏矩阵K
        K = dia_matrix((data, offsets), shape=(n,n)).tocsc()
        return K

    def bvec(self):
        '''假设rho(x)在每个单元内值为常数，仅对u_j(x)做积分'''
        return - np.sin(pi * self.x) * self.h

    def solve(self):
        k = self.Kmatrix()
        b = self.bvec()
        return inv(k) * b

    def compare(self):
        ground_truth = 1/(pi**2) * np.sin(pi * self.x)
        fem_res = self.solve()

        plt.plot(self.x, ground_truth, label="ground truth")
        plt.plot(self.x, fem_res, label="finite element")
        plt.title("number of nodes = %s"%self.nodes)
        plt.xlabel("x")
        plt.ylabal(r"$\phi(x)&")
        plt.legend(loc='best')
        plt.show()

fem = FEM(1001)
fem.compare()
