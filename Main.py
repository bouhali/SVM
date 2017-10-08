from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy, pylab, random, math

from SVMTools import SVMTools
from TestData import TestData


def main():
    # Create some test data
    test_generator = TestData()
    test_generator.gen_plot()
    test_data = test_generator.data
    # Separate the generated test data into x and t lists
    x = [[x, y] for x,y,_ in test_data]
    t = [sign for _,_,sign in test_data]
    assert(len(x) is len(t))

    # Set slack=0 to disable the use of slack variables
    svm_tools = SVMTools(x, t, kernel='polynomial', slack=50, polynomial_exp=3, radial_sigma=15)

    x_range = numpy.arange(-4, 4, 0.05)
    y_range = numpy.arange(-4, 4, 0.05)
    alpha_x_pairs = svm_tools.alpha_x_pairs
    grid = matrix([[svm_tools.indicator(alpha_x_pairs, [x, y])
        for y in y_range]
        for x in x_range])

    test_generator.add_contours(x_range, y_range, grid)

    test_generator.show_plot()


if __name__ == "__main__":
    main()
