F,N = map(int,raw_input().split())
#Train
X_train = []
Y_train = []
for row in xrange(N):
	data = map(float,raw_input().split())
	data_features = data[:F]
	price = data[F]
	X_train.append(data_features)
	Y_train.append(price)

import numpy as np
X_train = np.array(X_train)
Y_train = np.array(Y_train)

from sklearn.svm import SVR
linear = SVR(kernel = 'linear',C=1e5)
linear.fit(X_train,Y_train)

#Test data...predict using Polynomial Regression
t = input()
X_test = []
for row in xrange(t):
	data = map(float,raw_input().split())
	X_test.append(data)
X_test = np.array(X_test)

res = linear.predict(X_test)
for val in res:
	print("%.2f"%val)