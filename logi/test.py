from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

# 创建数据集
datasets = [
    ['青年', '否', '否', '一般', '否'],
    ['青年', '否', '否', '好', '否'],
    ['青年', '是', '否', '好', '是'],
    ['青年', '是', '是', '一般', '是'],
    ['青年', '否', '否', '一般', '否'],
    ['中年', '否', '否', '一般', '否'],
    ['中年', '否', '否', '好', '否'],
    ['中年', '是', '是', '好', '是'],
    ['中年', '否', '是', '非常好', '是'],
    ['中年', '否', '是', '非常好', '是'],
    ['老年', '否', '是', '非常好', '是'],
    ['老年', '否', '是', '好', '是'],
    ['老年', '是', '否', '好', '是'],
    ['老年', '是', '否', '非常好', '是'],
    ['老年', '否', '否', '一般', '否'],
]
labels = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']

# 转换为NumPy数组
datasets = np.array(datasets)

# 对数据集进行编码，将文本特征转换为数值特征
le = LabelEncoder()
for i in range(datasets.shape[1]):
    datasets[:, i] = le.fit_transform(datasets[:, i])

# 分离特征和标签
X = datasets[:, :-1]
y = datasets[:, -1]

# 创建决策树分类器
clf = DecisionTreeClassifier()

# 拟合（训练）决策树模型
clf.fit(X, y)

# 可视化决策树
fig, ax = plt.subplots(figsize=(12, 10))
tree.plot_tree(clf, feature_names=labels[:-1], class_names=le.classes_.tolist(), filled=True, ax=ax)
# 修改字体设置
font_properties = {'fontname': 'SimSun', 'fontsize': 25}  # 设置字体为宋体，字号为12
for text in ax.texts:
    text.set(**font_properties)

plt.show()