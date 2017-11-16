from read_data import train, test, mask
import numpy as np
from ACSM import ACSM
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

def predict_mvn(X, rv, min, max, N):
	y_pred = []
	for x in X:
		sum_prob = 0
		soma = 0
		for v in np.linspace(min, max, N):
			proba = rv.pdf(x[0:-1].tolist() + [v])
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

# *** item 1 ***

# *** MODELO ACSM ***
X, y, X_test, y_test = carregar_dataset(['peso','carga'], 'vo2max')
print("ACSM")
clf_acsm = ACSM()
y_pred = clf_acsm.predict(X_test)
print("\tmse:", clf_acsm.mse_error(y_test,y_pred))
print("\tscore:", clf_acsm.score(y_test,y_pred))


# *** GAUSSIANA MULTIVARIADA (peso,carga,idade,vo2max) ***
X, y, X_test, y_test = carregar_dataset(['peso','carga','idade', 'vo2max'], 'vo2max')
print("MVN[peso,carga,idade,vo2max]")
mu_x = X[:,0].mean()
mu_y = X[:,1].mean()
mu_z = X[:,2].mean()
mu_k = X[:,3].mean()
cov_matrix = np.cov(X.transpose())
rv = multivariate_normal([mu_x, mu_y, mu_z, mu_k], cov_matrix)
y_pred = predict_mvn(X_test, rv, 0, 100, 100)

print("\tPeso - Media:", mu_x)
print("\tCarga - Media:", mu_y)
print("\tIdade - Media:", mu_z)
print("\tVO2MAX - Media:", mu_k)
print("\tCov Mat:\n", cov_matrix)
print("\tmse:", mse_error(y_test, y_pred))
print("\tscore:", score(y_test, y_pred))

# *** item 3 ***
print("*** item 3 ***")
inits_idade = [30,50]
ends_idade = [40,60]
vo2s = [30, 20]
for i in range(len(inits_idade)):
	prob = 0.0
	for v_idade in np.linspace(inits_idade[i], ends_idade[i], 50):
		for v_peso in np.linspace(40, 180, 100):
			for v_carga in np.linspace(0, 500, 500):
				prob += rv.pdf([v_peso,v_carga,v_idade,vo2s[i]])
	print("\t(vo2max =",vo2s[i],") prob(",inits_idade[i],ends_idade[i],") =", round(prob, 5))


# *** item 4 ***
print("*** item 4 ***")
vo2 = 32.6
carg = 181
pes = 81.5
inits = [40, 50, 60]
ends = [50, 60, 70]

for i in range(len(inits)):
	prob = 0.0
	for v in np.linspace(inits[i], ends[i], 10000):
		prob += rv.pdf([pes,carg,v,vo2])
	print("prob(",inits[i],ends[i],") =", round(prob, 5))
