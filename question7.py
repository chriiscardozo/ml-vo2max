from sklearn.cluster import KMeans
from read_data import train, test, mask
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot_result(classes, titulo):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	for index, c in enumerate(classes):
		xs = c['X']
		ys = c['Y']
		zs = c['Z']
		ax.scatter(xs, ys, zs, c='C'+str(index), marker='.')
	ax.set_xlabel('idade')
	ax.set_ylabel('carga')
	ax.set_zlabel('vo2max')
	plt.title(titulo)
	plt.show()

def exec_kmeans(n):
	clf = KMeans(n_clusters=n,n_jobs=4,random_state=42)
	X = train[['idade','carga','vo2max']].as_matrix()

	y_pred = clf.fit_predict(X)

	clusters_ocorrencias = []
	classes = []
	for i in range(n):
		clusters_ocorrencias.append([0,0,0,0,0,0])
		classes.append({ 'X': [], 'Y': [], 'Z': [] })

	for index, x in enumerate(X):
		if(x[0] >= 18 and x[0] < 30):
			clusters_ocorrencias[y_pred[index]][0] += 1
		elif(x[0] >= 30 and x[0] < 50):
			clusters_ocorrencias[y_pred[index]][1] += 1
		elif(x[0] >= 50 and x[0] < 60):
			clusters_ocorrencias[y_pred[index]][2] += 1
		elif(x[0] >= 60 and x[0] < 70):
			clusters_ocorrencias[y_pred[index]][3] += 1
		elif(x[0] >= 70 and x[0] < 80):
			clusters_ocorrencias[y_pred[index]][4] += 1
		elif(x[0] >= 80 and x[0] < 100):
			clusters_ocorrencias[y_pred[index]][5] += 1

		classes[y_pred[index]]['X'].append(x[0])
		classes[y_pred[index]]['Y'].append(x[1])
		classes[y_pred[index]]['Z'].append(x[2])

	clusters_ocorrencias = np.array(clusters_ocorrencias)
	return (100.0*clusters_ocorrencias/len(X), classes)

freq, classes = exec_kmeans(3)
print("*** Freq. para K = 3")
print(freq)
for i in freq: print(sum(i))
plot_result(classes, "K = 3")

freq, classes = exec_kmeans(4)
print("*** Freq. para K = 4")
print(freq)
for i in freq: print(sum(i))
plot_result(classes, "K = 4")