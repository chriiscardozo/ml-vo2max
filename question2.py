import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from read_data import train, test, mask
from ACSM import ACSM

# *** item 1 ***
print("*** Item 1 ***")
print("Usando X=[carga,vo2max]")
X = train[['carga','vo2max']].as_matrix()
X1 = X[:,0]
X2 = X[:,1]

#Parameters to set
mu_x = X1.mean()
variance_x = X1.std()**2.0

mu_y = X2.mean()
variance_y = X2.std()**2.0

cov_matrix = np.cov(X.transpose())

print("Carga\n\tMedia:", mu_x, "| variancia:", variance_x)
print("VO2Max\n\tMedia:", mu_y, "| variancia:", variance_y)

print("Matriz covariancia:\n", cov_matrix)

#Create grid and multivariate normal
x = np.linspace(X1.min(),X1.max(),100)
y = np.linspace(X2.min(),X2.max(),100)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], cov_matrix)

#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
ax.set_xlabel('carga')
ax.set_ylabel('vo2max')
ax.set_zlabel('pdf')
plt.show()

# *** item 2 ***
print("*** item 2 ***")
print("Usando X=[peso,carga,vo2max]")
X = train[['peso','carga','vo2max']].as_matrix()
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

# [peso, carga] = vo2max
# [73 325] = 57.6712328767
# [80.3 60] = 10.3362391034
# [89 310] = 52.4719101124

X = [[73., 325.], [80.3, 60.0], [89., 310.]]
y = [57.6712328767, 10.3362391034, 52.4719101124]

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

y_pred = predict_mvn(X, rv, X3.min(), X3.max(), 1000)
mse = ((np.array(y) - np.array(y_pred)) ** 2).sum() / len(y)
print("Predict com MVN")
print("\tX =", X)
print("\tY =", y)
print("\ty_pred (MVN) =", y_pred)
print("\tmse (MVN) = ", mse)

modelo = ACSM()
y_pred = modelo.predict(X)
mse = modelo.mse_error(y, y_pred)
print("\ty_pred (ACSM) =", y_pred)
print("\tmse (ACSM) =", mse)


# *** item 3 ***
print("*** item 3 ***")
print("Usando X=[idade,peso,carga,vo2max]")
X = train[['idade','peso','carga','vo2max']].as_matrix()
X1 = X[:,0]
X2 = X[:,1]
X3 = X[:,2]
X4 = X[:,3]

#Parameters to set
mu_x = X1.mean()
variance_x = X1.std()**2.0

mu_y = X2.mean()
variance_y = X2.std()**2.0

mu_z = X3.mean()
variance_z = X3.std()**2.0

mu_k = X4.mean()
variance_k = X4.std()**2.0

cov_matrix = np.cov(X.transpose())

print("Idade\n\tMedia:", mu_x, "| variancia:", variance_x)
print("Peso\n\tMedia:", mu_y, "| variancia:", variance_y)
print("Carga\n\tMedia:", mu_z, "| variancia:", variance_z)
print("VO2Max\n\tMedia:", mu_k, "| variancia:", variance_k)

print("Matriz covariancia:\n", cov_matrix)

rv = multivariate_normal([mu_x, mu_y, mu_z, mu_k], cov_matrix)

# [idade, peso, carga] = vo2max
# [31 73 325] = 57.6712328767
# [66 80.3 60] = 10.3362391034
# [40 89 310] = 52.4719101124

X = [[31, 73., 325.], [66, 80.3, 60.0], [40, 89., 310.]]
y = [57.6712328767, 10.3362391034, 52.4719101124]

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

y_pred = predict_mvn(X, rv, X4.min(), X4.max(), 1000)
mse = ((np.array(y) - np.array(y_pred)) ** 2).sum() / len(y)
print("Predict com MVN")
print("\tX =", X)
print("\tY =", y)
print("\ty_pred (MVN) =", y_pred)
print("\tmse (MVN) = ", mse)