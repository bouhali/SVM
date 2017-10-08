from cvxopt.solvers import qp
from cvxopt.base import matrix, spdiag, sparse
import numpy, pylab, random, math



print(random.normalvariate(-1.5, 1))

class SVM():

	def __init__(self, data, target):
		self.data = matrix(data).trans()
		self.target = matrix(target)
		self.Pmatrix = matrix(0, (self.data.size[0], self.data.size[0]), 'd')
		self.q = matrix(-1, (self.data.size[0],1),'d')
		#print(self.q)
		self.h = matrix(0, (self.data.size[0],1), 'd')
		#print(self.h)
		self.G = spdiag(matrix(-1, (1, self.data.size[0]), 'd'))
		self.alphas = []
		self.alphas_x = []
		self.G_slack = matrix([self.G, spdiag(matrix(1, (1, self.data.size[0]), 'd'))])
		self.h_slack = matrix([self.h, matrix(1, (self.data.size[0],1), 'd')] )


	def creatPmatrix(self):
		for i in range(self.data.size[0]):
			for j in range(self.data.size[0]):
				#print(self.Pmatrix)
				self.Pmatrix[i,j] = self.target[i]*self.target[j]*self.polKernel(self.data[i,:], self.data[j,:], 2)

	def getPmatrix(self):
		print(self.Pmatrix)


	def linKernel(self, x, y):
		#print((x.trans()*y) + 1)
		return (x*y.trans())[0]+1

	def polKernel(self, x, y, p):
		#print((x.trans()*y[0] +1)**p)
		return ((x*y.trans())[0] +1)**p

	def radKernel(self, x, y, s):
		return math.e**(-(x-y).trans()*(x-y))[0]/(2*s**2)

	def findAlpha(self):
		r = qp(self.Pmatrix, self.q, self.G_slack, self.h_slack)
		self.alphas = list(r['x'])

	def ZeroAlpha_Xdata(self):
		for i in range(len(self.alphas)):
			if self.alphas[i] > 10e-5:
				self.alphas_x.append((self.data[i,:], self.target[i], self.alphas[i]))

	def indicator(self, newX):
		indSum = 0
		for x, t, a in self.alphas_x:
			indSum += a*t*self.polKernel(newX, x, 2)
		return indSum

#def main():
#	x=[[1,2], [3,3], [2,1]]
#	y=[1,-1,1]
#	mysvm=SVM(x,y)
#	mysvm.creatPmatrix()
#	mysvm.getPmatrix()
#	mysvm.findAlpha()
#	print(mysvm.alphas)
#	mysvm.ZeroAlpha_Xdata()
#	print(mysvm.alphas_x)
#	print(matrix([1,1]).trans().size)
#	print(mysvm.indicator(matrix([1,1]).trans()))
#	#print(mysvm.data)
#	#print(mysvm.linKernel(mysvm.data[0,:],mysvm.data[1,:]))
#	#print(spdiag(matrix(-1, (2,1))))
#
#
#if __name__ == "__main__":
#	main()

#A = matrix([[1,2], [3,3]])
#print(A.size)
#print(A)
#print(A[0,:].trans().size)
#print(A[1,:].size)

