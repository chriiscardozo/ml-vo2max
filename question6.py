from read_data import train, test, mask
import numpy as np
from Regression import Regression
from ACSM import ACSM
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

def predict_mvn(X, rv, min, max, N):
	y_pred = []
	for x in X:
		sum_prob = 0
		soma = 0
		for v in np.linspace(min, max, N):
			proba = rv.pdf([x[0],x[1], x[2], v])
			soma += (v * proba)
			sum_prob += proba
		y_pred.append(soma/sum_prob)
	return y_pred

def score(y_true, y_pred):
	residual = ((y_true - y_pred) ** 2).sum()

	media_y_true = [sum(y_true) / float(len(y_true))] * len(y_true)
	soma_quadrados = ((np.array(y_true) - np.array(media_y_true)) ** 2).sum()
	return 1 - (residual/soma_quadrados)

def mse_error(y_true, y_pred):
		return ((np.array(y_true) - np.array(y_pred)) ** 2).sum() / len(y_true)

def carregar_dataset(X_fields, y_field):
	X = train[X_fields].as_matrix()
	y = train[y_field].as_matrix()

	X_test = test[X_fields].as_matrix()
	y_test = test[y_field].as_matrix()

	return [X, y, X_test, y_test]

X, y, X_test, y_test = carregar_dataset(['peso','carga'], 'vo2max')

# *** MODELO ACSM ***
print("ACSM")
clf_acsm = ACSM()
y_pred = clf_acsm.predict(X_test)
print("\tmse:", clf_acsm.mse_error(y_test,y_pred))
print("\tscore:", clf_acsm.score(y_test,y_pred))

# *** MODELO REGRESSÃO D=2 VARS INDEPENDENTES ***
print("Regressão 1 (sem dep)")
modelo = Regression()
modelo.phi_inter = lambda X, d: np.array([[1, x[0], x[1], x[0]**2, x[1]**2] for x in X])
modelo.fit(X, y)
y_pred = modelo.predict(X_test)

print("\tmse:", modelo.mse_error(y_test, y_pred))
print("\tscore:", modelo.score(y_test, y_pred))

# *** MODELO REGRESSÃO D=2 DEPENDENCIA X*Y ***
print("Regressão 2 (com dep)")
modelo = Regression()
modelo.phi_inter = lambda X, d: np.array([[1, x[0], x[1], x[0]*x[1], x[0]**2, x[1]**2, (x[0]**2)*x[1], (x[1]**2)*x[0], (x[0]**2)*(x[1]**2), x[0]/x[1]] for x in X])
modelo.fit(X, y)
y_pred = modelo.predict(X_test)

print("\tmse:", modelo.mse_error(y_test, y_pred))
print("\tscore:", modelo.score(y_test, y_pred))


X, y, X_test, y_test = carregar_dataset(['peso','carga','idade'], 'vo2max')

# *** Regressão 3 vars D=3 com dep ***
print("Regressão 3 (3 vars c/ dep)")
modelo = Regression()
modelo.phi_inter = lambda X, d: np.array([[1, x[0], x[1], 1/x[2], x[0]*x[1], x[0]**2, x[1]**2, (x[0]**2)*x[1], (x[1]**2)*x[0], (x[0]**2)*(x[1]**2), x[0]/x[1]] for x in X])
modelo.fit(X, y)
y_pred = modelo.predict(X_test)

print("\tmse:", modelo.mse_error(y_test, y_pred))
print("\tscore:", modelo.score(y_test, y_pred))

X, y, X_test, y_test = carregar_dataset(['peso','carga','idade', 'vo2max'], 'vo2max')

# *** GAUSSIANA MULTIVARIADA ***
print("Gaussiana Multivariada")
#Parameters to set
mu_x = X[:,0].mean()
mu_y = X[:,1].mean()
mu_z = X[:,2].mean()
mu_k = X[:,3].mean()
cov_matrix = np.cov(X.transpose())

rv = multivariate_normal([mu_x, mu_y, mu_z, mu_k], cov_matrix)

print("\tPeso - Media:", mu_x)
print("\tCarga - Media:", mu_y)
print("\tIdade - Media:", mu_z)
print("\tVO2MAX - Media:", mu_k)
print("\tCov Mat:\n", cov_matrix)

y_pred = predict_mvn(X_test, rv, 0, 100, 100)
print("\tmse:", mse_error(y_test, y_pred))
print("\tscore:", score(y_test, y_pred))