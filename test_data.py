from cvxopt.solvers import qp
from cvxopt.base import matrix, spdiag
import numpy, pylab, random, math
from SVMcode import SVM

#numpy.random.seed(100)
classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1)  for i in range(5)] +\
[(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1)  for i in range(5)]


classB = [(random.normalvariate(0, 0.5), random.normalvariate(-0.5, 0.5), -1)  for i in range(10)]


data = classA + classB

random.shuffle(data)


xData = []
yData = []
for x in data:
	xData.append([x[0], x[1]])
	yData.append(x[2])



mysvm = SVM(xData,yData)
mysvm.creatPmatrix()
mysvm.findAlpha()
mysvm.ZeroAlpha_Xdata()
xxrange = numpy.arange(-4,4,0.05)
yyrange = numpy.arange(-4,4,0.05)


grid = matrix([[mysvm.indicator(matrix([x,y]).trans()) for y in yyrange] for x in xxrange])

pylab.hold(True)
pylab.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
pylab.plot([p[0] for p in classB], [p[1] for p in classB], 'ro')
pylab.contour(xxrange, yyrange, grid, (-1,0,1), colors = ('red', 'black', 'blue'), linewidths = (1,3,1))
pylab.show()


