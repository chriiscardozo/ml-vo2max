from read_data import train, test, mask
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.naive_bayes import GaussianNB

# *** item 1 ***
print("*** item 1 ***")
print("Usando X=[peso,carga,vo2max]")
X_M1 = (train[(train['idade']>=18) & (train['idade']<40)])[['peso', 'carga', 'vo2max']].as_matrix()
X_M2 = (train[(train['idade']>=40) & (train['idade']<60)])[['peso', 'carga', 'vo2max']].as_matrix()
X_M3 = (train[(train['idade']>=60)])[['peso', 'carga', 'vo2max']].as_matrix()

X_M1_test = (test[(test['idade']>=18) & (test['idade']<40)])[['peso', 'carga', 'vo2max']].as_matrix()
X_M2_test = (test[(test['idade']>=40) & (test['idade']<60)])[['peso', 'carga', 'vo2max']].as_matrix()
X_M3_test = (test[(test['idade']>=60)])[['peso', 'carga', 'vo2max']].as_matrix()

for index, data in enumerate([(X_M1, X_M1_test), (X_M2,X_M2_test), (X_M3,X_M3_test)]):
	print("< Modelo", index+1, ">")
	X, X_test = data
	X1 = X[:,0]
	X2 = X[:,1]
	X3 = X[:,2]

	#Parameters to set
	mu_x = X1.mean()
	variance_x = X1.std()**2.0

	mu_y = X2.mean()
	variance_y = X2.std()**2.0

	mu_z = X3.mean()
	variance_z = X3.std()**2.0

	cov_matrix = np.cov(X.transpose())

	print("Peso\n\tMedia:", mu_x, "| variancia:", variance_x)
	print("Carga\n\tMedia:", mu_y, "| variancia:", variance_y)
	print("VO2Max\n\tMedia:", mu_z, "| variancia:", variance_z)

	print("Matriz covariancia:\n", cov_matrix)

	rv = multivariate_normal([mu_x, mu_y, mu_z], cov_matrix)

	def predict_mvn(X, rv, min, max, N):
		y_pred = []
		for x in X:
			sum_prob = 0
			soma = 0
			for v in np.linspace(min, max, N):
				proba = rv.pdf([x[0],x[1], v])
				soma += (v * proba)
				sum_prob += proba
			y_pred.append(soma/sum_prob)
		return y_pred

	X = X_test[:,:2]
	y = X_test[:, 2]

	y_pred = predict_mvn(X, rv, X3.min(), X3.max(), 1000)
	mse = ((np.array(y) - np.array(y_pred)) ** 2).sum() / len(y)
	print("Predict com MVN")
	print("\tX =", X)
	print("\tY =", y)
	print("\ty_pred (MVN) =", y_pred)
	print("\tmse (MVN) = ", mse)

# *** item 5 ***
print("*** Item 5 ***")
X_M1 = (train[(train['idade']>=18) & (train['idade']<40)])[['peso', 'carga', 'vo2max']].as_matrix()
X_M2 = (train[(train['idade']>=40) & (train['idade']<60)])[['peso', 'carga', 'vo2max']].as_matrix()
X_M3 = (train[(train['idade']>=60)])[['peso', 'carga', 'vo2max']].as_matrix()

X_M1_test = (test[(test['idade']>=18) & (test['idade']<40)])[['peso', 'carga', 'vo2max']].as_matrix()
X_M2_test = (test[(test['idade']>=40) & (test['idade']<60)])[['peso', 'carga', 'vo2max']].as_matrix()
X_M3_test = (test[(test['idade']>=60)])[['peso', 'carga', 'vo2max']].as_matrix()

y = [0] * X_M1.shape[0]
y.extend([1] * X_M2.shape[0])
y.extend([2] * X_M3.shape[0])

y_test = [0] * X_M1_test.shape[0]
y_test.extend([1] * X_M2_test.shape[0])
y_test.extend([2] * X_M3_test.shape[0])

X = np.concatenate((X_M1, X_M2, X_M3))
X_test = np.concatenate((X_M1_test, X_M2_test, X_M3_test))

gnb = GaussianNB()
gnb.fit(X, y)
y_pred = gnb.predict(X_test)

print("y_pred =", y_pred)
print("score =", gnb.score(X_test, y_test))