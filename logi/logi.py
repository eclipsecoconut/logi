import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
filename = 'Folds5x2_pp.csv'
data = np.genfromtxt(filename, delimiter=',', dtype=float)
data = data[1:]
X = []
Y = []
for each_data in data:
    X.append(each_data[:-1])
    Y.append(each_data[-1])
# print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
reg = LinearRegression().fit(X_train, Y_train)
yhat = reg.predict(X_test)
yhat_train = reg.predict(X_train)
print('训练集的mse = {}'.format(mean_squared_error(yhat_train, Y_train)))
print('测试集的mse = {}'.format(mean_squared_error(yhat,Y_test)))
plt.figure()
plt.xlabel('Measured')
plt.ylabel('Predict')
plt.scatter(Y_test, yhat)
plt.show()
a = input("数据输入：")
a = [[float(s) for s in a.split(",")]]
print(reg.predict(a))