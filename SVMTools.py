from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy, pylab, random, math


class SVMTools():
    def __init__(self, x, t):
        self.train(x, t)

    def kernel(self, x, y):
        # Linear
        return numpy.dot(x, y) + 1

    def p(self, t, x):
        N = len(t)
        P = numpy.zeros((N, N))
        for i in range(0, N):
            for j in range(0, N):
                P[i][j] = t[i] * t[j] * self.kernel(x[i], x[j])
        return P

    def indicator(self, alpha_x_pairs, new_x):
        sum = 0
        for x, t, alpha in alpha_x_pairs:
            sum += alpha * t * self.kernel(new_x, x)
        return numpy.sign(sum)
    
    def train(self, x, t):
        N = len(x)
        assert(N is len(t))

        q = numpy.zeros(N)
        for i in range(0, N):
            q[i] = -1

        h = numpy.zeros(N)

        G = numpy.zeros((N, N))

        for i in range(0, N):
            G[i][i] = -1

        P = self.p(t, x)

        r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
        alphas = list(r['x'])
        # print(alphas)
        
        # Extract non-zero alphas with data points
        self.alpha_x_pairs = []
        for i in range(0, N):
            if alphas[i] > 10e-5:
                self.alpha_x_pairs.append((x[i], t[i], alphas[i]))
        # print(alpha_x_pairs)


    def classify(self, new_x):
        return self.indicator(self.alpha_x_pairs, new_x)


def test():
    x = numpy.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [0, 1],
        [0, 2],
        [0, 3]
        ])
    t = numpy.array([
        1 , 1, 1, -1, -1, -1
        ])

    svm = SVMTools(x, t)

    new_x = [4, 0]
    new_classification = svm.classify(new_x)
    print(new_classification)


if __name__ == "__main__":
    test()
