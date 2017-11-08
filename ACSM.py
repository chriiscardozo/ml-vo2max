import numpy as np

class ACSM:
	def foo(self, carga, peso):
		return (carga*11.4 + 260 + peso*3.5)/peso

	def predict(self, X):
		return [ self.foo(x[1],x[0]) for x in X] 

	def mse_error(self, y_true, y_pred):
		return ((np.array(y_true) - np.array(y_pred)) ** 2).sum() / len(y_true)

	def score(self, y_true, y_pred):
		residual = ((y_true - y_pred) ** 2).sum()

		media_y_true = [sum(y_true) / float(len(y_true))] * len(y_true)
		soma_quadrados = ((np.array(y_true) - np.array(media_y_true)) ** 2).sum()
		return 1 - (residual/soma_quadrados)