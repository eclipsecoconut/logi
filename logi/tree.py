import math


def create_data():  # 创造示例数据
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
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    # 返回数据集和每个维度的名称
    return datasets, labels


'''计算熵'''


def calc_entropy(dataSet):
    numDataSet = len(dataSet)
    typeCount = {}
    for row in dataSet:
        if row[-1] not in typeCount.keys():
            typeCount[row[-1]] = 0
        typeCount[row[-1]] += 1
    print('每类样本出现个数', typeCount)
    entropy = 0.0
    for num in typeCount.values():
        p = num / numDataSet
        entropy -= p * math.log(p, 2)
    print('entropy =', entropy)
    return entropy


'''选择最优的分类特征'''


def choose_feature(dataSet):
    numFeatures = len(dataSet[0]) - 1
    entropy0 = calc_entropy(dataSet)
    bestGain = 0  # 信息增益
    bestIndex = -1  # 最优特征下标
    for i in range(numFeatures):
        featList = [row[i] for row in dataSet]
        uniqueFeatValues = set(featList)
        new = 0
        for value in uniqueFeatValues:
            subDataSet = splitDataSet(dataSet, i, value)
            print('划分后的子集 :', subDataSet)
            weight = len(subDataSet) / float(len(dataSet))
            new += weight * calc_entropy(subDataSet)
        print('划分后的信息熵 :', new)
        gain = entropy0 - new
        if gain > bestGain:
            bestGain = gain
            bestIndex = i
    return bestIndex


'''
划分数据集
axis : 最优特征BestFeature(BF)所在下标
value : BF能取得值
'''


def splitDataSet(dataSet, axis, value):  # 按某个特征分类后的数据
    retDataSet = []
    for row in dataSet:
        if row[axis] == value:
            reducedFeatvec = row[:axis]  # 取出分裂特征前的数据集
            reducedFeatvec.extend(row[axis + 1:])  # 取出分裂特征后的数据集,合并两部分数据集
            retDataSet.append(reducedFeatvec)  # 本行取得的去除value的列表 加入总列表
    return retDataSet


'''统计，多者胜出'''


def majorityCnt(typeList):  # 按分类后类别数量排序，比如：最后分类为2男1女，则判定为男；
    typeCount = {}
    for t in typeList:
        if t not in typeCount.keys():
            typeCount[t] = 0
        typeCount[t] += 1
    print('typeCount =', typeCount)
    sortedTypeCount = sorted(typeCount.items(), key=lambda x: x[1], reverse=True)  # 从大到小排列，结果如[('女', 2), ('男', 1)]
    print('少数服从多数，多数为 :', sortedTypeCount[0][0])
    return sortedTypeCount[0][0]


'''递归建树'''


def createTree(dataSet, labels):
    typeList = [row[-1] for row in dataSet]  # 类别：男或女
    if typeList.count(typeList[0]) == len(typeList):  # 若只有一个类，直接返回
        return typeList[0]
    if len(dataSet[0]) == 1:  # 若最后只剩下一个类别属性
        return majorityCnt(typeList)
    bestFeatIndex = choose_feature(dataSet)  # 最优特征下标和对应特征
    bestFeat = labels[bestFeatIndex]
    print('bestFeatureIndex =', bestFeatIndex)
    print('***********最优特征值 =', bestFeat, end='***********\n')

    myTree = {bestFeat: {}}  # 分类结果以字典形式保存
    del (labels[bestFeatIndex])

    uniqueVals = set()  # 最优特征能取的值，用set保证无重复
    {uniqueVals.add(row[bestFeatIndex]) for row in dataSet}
    print(f'{bestFeat} 能取的值 :', uniqueVals)
    for value in uniqueVals:
        subLabels = labels  # labels里已经删去了最优特征，用subLabels为了区分更明显
        myTree[bestFeat][value] = \
            createTree(splitDataSet(dataSet, bestFeatIndex, value), subLabels)
    return myTree


if __name__ == '__main__':
    dataset, labels = create_data()
    print(createTree(dataset, labels))  # 输出决策树模型结果
