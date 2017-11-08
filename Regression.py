import numpy as np
import matplotlib.pyplot as plt

class Regression:
	def __init__(self, degree=1):
		self.W = None
		self.degree = degree

	def _phi(self, X, d):
		return [[1] + np.asarray([[np.power(v, p+1) for p in range(d)] for v in x]).flatten().tolist() for x in X]

	def fit(self, X, y):
		_X = self._phi(X, self.degree)

		_X_t = np.transpose(_X)
		inversa = np.linalg.inv(np.dot(_X_t, _X))
		X_adaga = np.dot(inversa, _X_t)
		self.W = np.dot(X_adaga, y)
		return self.W

	def predict(self, X):
		_X = self._phi(X, self.degree)
		if(self.W is None):
			raise Exception('W value not initilized. Use fit before predict.')
		else:
			return np.transpose(np.dot(self.W, np.transpose(_X)))

	def error_mse(self, y_true, y_pred):
		return ((np.array(y_true) - np.array(y_pred)) ** 2).sum() / len(y_true)

	def score(self, y_true, y_pred):
		residual = ((y_true - y_pred) ** 2).sum()

		media_y_true = [sum(y_true) / float(len(y_true))] * len(y_true)
		soma_quadrados = ((np.array(y_true) - np.array(media_y_true)) ** 2).sum()
		return 1 - (residual/soma_quadrados)

# exemplo
def main():
	X = [[0.5],[0.3],[0.4],[0.45],[0.25]]
	y = [5.0, 2.7, 4.4, 4.0, 2.0]

	clf = Regression()
	W = clf.fit(X, y)
	y_pred = clf.predict(X)

	print('W =', W)
	print('pred =', y_pred)

	score = clf.score(y, y_pred)
	print('Score:', score)
	plt.plot([x[0] for x in X], y, 'go')
	plt.plot([x[0] for x in X], y_pred, 'rx')

	x_plot = [[1]+[x] for x in np.linspace(0.2,0.6,100)]
	y_plot = np.dot(x_plot, W)
	plt.plot([x[1] for x in x_plot] ,y_plot, 'b-')

	plt.show()

if __name__ == '__main__':
	main()