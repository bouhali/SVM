from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy, pylab, random, math


class SVMTools():
    def __init__(self, x, t, kernel='polynomial', slack=True,  polynomial_exp=2, radial_sigma=1, sigmoid_k=1/2, sigmoid_delta=5):
        self.kernel_type = kernel
        self.slack = slack
        self.polynomial_exp = polynomial_exp
        self.radial_sigma = radial_sigma
        self.sigmoid_k = sigmoid_k
        self.sigmoid_delta = sigmoid_delta
        self.train(x, t)

    def kernel(self, x, y):
        if self.kernel_type == 'linear':
            x = numpy.transpose(x)
            return numpy.dot(x, y) + 1

        elif self.kernel_type == 'polynomial':
            x = numpy.transpose(x)
            return numpy.power((numpy.dot(x, y) + 1), self.polynomial_exp)

        elif self.kernel_type == 'radial':
            w = numpy.subtract(x, y)
            w = numpy.dot(w, w)
            return numpy.exp(-w / (2*numpy.power(self.radial_sigma, 2)))

        elif self.kernel_type == 'sigmoid':
            # raise Exception('TODO not implemented correctly')
            x = numpy.transpose(x)
            # TODO + or - delta???
            return numpy.tanh(
                    numpy.dot( numpy.multiply(self.sigmoid_k, x), y ) + self.sigmoid_delta
                    # self.sigmoid_k * numpy.dot(x, y) + self.sigmoid_delta
                    )
        else:
            raise ValueError('Kernel not available')

    def p(self, t, x):
        N = len(t)
        P = numpy.zeros((N, N))
        for i in range(0, N):
            for j in range(0, N):
                P[i][j] = t[i] * t[j] * self.kernel(x[i], x[j])
        return P

    def indicator(self, alpha_x_pairs, new_x):
        sum = 0
        for xy, t, alpha in alpha_x_pairs:
            sum += alpha * t * self.kernel(new_x, xy)
        return sum
    
    def train(self, x, t):
        N = len(x)
        assert(N is len(t))

        q = numpy.zeros(N)
        for i in range(0, N):
            q[i] = -1

        if self.slack:
            h = numpy.zeros((2*N, 1))
            for i in range(N, 2*N):
                h[i] = self.slack
            # print(h)
        else:
            h = numpy.zeros(N)

        if self.slack:
            # TODO fix G
            G = numpy.array([[0.0 if y >= N else 0.0 for x in range(N)] for y in range(2*N)])
        else:
            G = numpy.zeros((N, N))

        if self.slack:
            for i in range(0, N):
                G[i][i] = -1.0
                # Lower part of G
                G[N+i][i] = 1.0
        else:
            for i in range(0, N):
                G[i][i] = -1.0
        # print(G)

        P = self.p(t, x)

        r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
        alphas = list(r['x'])
        # print(alphas)
        
        # Extract non-zero alphas with data points
        self.alpha_x_pairs = []
        for i in range(0, N):
            if alphas[i] > 10e-5:
                self.alpha_x_pairs.append((x[i], t[i], alphas[i]))
        print('Number of support vectors: ', len(self.alpha_x_pairs))


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
