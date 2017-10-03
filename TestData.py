from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy, pylab, random, math


class TestData():
    def __init__(self):
        self.data = self.generate()

    def generate(self):
        # numpy.random.seed(100)
        self.classA = [(random.normalvariate(-1.5, 1),
            random.normalvariate(0.5, 1),
            1.0)
            for i in range(5)] + \
                    [(random.normalvariate(1.5, 1),
                        random.normalvariate(0.5, 1),
                        1.0)
                        for i in range(5)]

        self.classB = [(random.normalvariate(0.0, 0.5),
            random.normalvariate(-0.5, 0.5),
            -1.0)
            for i in range(10)]
        self.data = self.classA + self.classB
        random.shuffle(self.data)

    def plot(self):
        pylab.hold(True)
        pylab.plot(
                [p[0] for p in self.classA],
                [p[1] for p in self.classA],
                'bo'
                )
        pylab.plot(
                [p[0] for p in self.classB],
                [p[1] for p in self.classB],
                'ro'
                )
        pylab.show()

if __name__ == "__main__":
    td = TestData()
    data = td.data
    td.plot()
