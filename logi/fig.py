import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
filename = 'mnist-train.csv'
data = np.genfromtxt(filename, delimiter=',', dtype=float)
data = data[1:]
X = []
Y = []
for each_data in data:
    X.append(each_data[1:])
    Y.append(each_data[0])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)
Y_pre = model.predict(X_test)
# 计算混淆矩阵
cm = confusion_matrix(Y_test, Y_pre)
print("Confusion Matrix:")
print(cm)
# 提取混淆矩阵中的TP、FP、TN、FN
TP = cm[1, 1]
FP = cm[0, 1]
TN = cm[0, 0]
FN = cm[1, 0]
# 计算准确率
accuracy = accuracy_score(Y_test, Y_pre)
print("Accuracy:", accuracy)
