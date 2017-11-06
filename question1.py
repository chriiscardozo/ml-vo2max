from read_data import train, test, mask
import numpy as np


#function to apply phi function
def apply_phi(X, func, d):
	return [func(x,d) for x in X]

#function to calculate the best W to the model normal
def fit_linear_regretion(X_in, y_in):
	X = np.matrix(X_in)
	y = np.matrix(y_in).transpose()
	return np.linalg.inv(X.transpose() * X) * X.transpose() * y

#calculate the y value applying W
def apply_w(X, phi, w):
	y = lambda x: np.array(phi(x))*np.array(w)
	return [ y(x) for x in X ]


#generate phi function
phi = lambda x, d: [1] + np.matrix([ np.power(x,i+1) for i in range(d)]).flatten().tolist()[0]


# *** item 1 ***
# X = train[['carga','idade']]
X = train['carga'].as_matrix()
y = train['vo2max'].as_matrix()

X_test = test['carga'].as_matrix()
y_test = test['vo2max'].as_matrix()

# caculate the values of w
w = fit_linear_regretion(apply_phi(X, phi, 5), y)

print("w: ",w)

# *** item 2 ***
#X = train[['peso','carga']].as_matrix()
#y = train['vo2max'].as_matrix()

#X_test = test['carga'].as_matrix()
#y_test = test['vo2max'].as_matrix()

# caculate the values of w
#w = fit_linear_regretion(apply_phi(X, phi), y)

#print("w: ",w)
