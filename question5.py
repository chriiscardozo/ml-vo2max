from sklearn.mixture import GaussianMixture
from read_data import train, test, mask
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_result(classes, titulo):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	for index, c in enumerate(classes):
		xs = c['X']
		ys = c['Y']
		zs = c['Z']
		ax.scatter(xs, ys, zs, c='C'+str(index), marker='.')
	ax.set_xlabel('peso')
	ax.set_ylabel('carga')
	ax.set_zlabel('vo2max')
	plt.title(titulo)
	plt.show()


def get_gabarito_esperado():
	# Plot do gráfico esperado se idade fosse um cluster
	X_M1 = (train[(train['idade']>=18) & (train['idade']<40)])[['peso', 'carga', 'vo2max']].as_matrix()
	X_M2 = (train[(train['idade']>=40) & (train['idade']<60)])[['peso', 'carga', 'vo2max']].as_matrix()
	X_M3 = (train[(train['idade']>=60)])[['peso', 'carga', 'vo2max']].as_matrix()
	c_1 = { 'X': X_M1[:,0],
			'Y': X_M1[:,1],
			'Z': X_M1[:,2]
		  }
	c_2 = { 'X': X_M2[:,0],
			'Y': X_M2[:,1],
			'Z': X_M2[:,2]
		  }
	c_3 = { 'X': X_M3[:,0],
			'Y': X_M3[:,1],
			'Z': X_M3[:,2]
		  }
	return [c_1, c_2, c_3]

def get_gaussiana_cluster(data, n):
	clf = GaussianMixture(n, init_params='random')

	clf.fit(data)
	y_pred = clf.predict(data)

	classes = []
	for i in range(n):
		classes.append({ 'X': [], 'Y': [], 'Z': [] })

	for index, x in enumerate(data):
		classes[y_pred[index]]['X'].append(x[0])
		classes[y_pred[index]]['Y'].append(x[1])
		classes[y_pred[index]]['Z'].append(x[2])

	return (clf, classes)


X = train[['peso', 'carga','vo2max']].as_matrix()

plot_result(get_gabarito_esperado(), 'Clusters por faixa etária da q4')

for n in range(2, 5):
	print("=> EM: K =", n)
	clf, result = get_gaussiana_cluster(X, n)
	print("Medias: ", clf.means_)
	print("Covariancias: ", clf.covariances_)
	print("Aprioris:", clf.weights_)
	print("Negative Log-likelihood:", clf.lower_bound_)
	plot_result(result, 'Clusters EM: K = ' + str(n))
