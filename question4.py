from read_data import train, test, mask
from sklearn.mixture import GaussianMixture

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

clf = GaussianMixture(n_components=3)
clf.fit(X)
y_pred = clf.predict(X_test)