from read_data import train, test, mask
import numpy as np
from Regression import Regression
from ACSM import ACSM
from sklearn.mixture import GaussianMixture

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
modelo.phi_inter = lambda X, d: np.array([[1, x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1], (x[0]**2)*x[1], (x[1]**2)*x[0], (x[0]**2)*(x[1]**2)] for x in X])
modelo.fit(X, y)
y_pred = modelo.predict(X_test)

print("\tmse:", modelo.mse_error(y_test, y_pred))
print("\tscore:", modelo.score(y_test, y_pred))