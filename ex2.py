# 逻辑回归、正则化
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import scipy.optimize as opt

# 1 逻辑回归
# 1.1 数据可视化
path = 'ex2_data/ex2data1.txt';
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
# .isin：用来清洗数据，删选过滤掉DataFrame中一些行
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()


# 1.2 实现
# 1.2.1 sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z));

# 1.2.2 代价函数和梯度
def cost(theta, X, y):
    theta = np.matrix(theta);
    X = np.matrix(X);
    y = np.matrix(y);
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

data.insert(0, 'Ones', 1)
cols = data.shape[1];
X = data.iloc[:, 0:cols - 1];
y = data.iloc[:, cols - 1:cols];
theta = np.zeros(3)
X = np.array(X.values);
y = np.array(y.values)
result = cost(theta, X, y)

# 梯度计算函数 并未更新theta
def gradient(theta, X, y):
    theta = np.matrix(theta);
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad

# 1.2.3 用工具库计算theta的值
# (array([-25.16131872,   0.20623159,   0.20147149]), 36, 0)
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
# 用theta的计算结果代回代价函数计算
# 0.20349770158947425
resultx = cost(result[0], X, y)
# 画出决策曲线
'''
plotting_x1 = np.linspace(30, 100, 100)
plotting_h1 = (- result[0][0] - result[0][1] * plotting_x1) / result[0][2]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(plotting_x1, plotting_h1, 'y', label="Prediction")
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()
'''

# 1.2.4 评价逻辑回归模型
# 实现h_theta
def hfunc1(theta, X):
    return sigmoid(np.dot(theta.T, X))

# 0.776290625526598
hfunc1(result[0], [1, 45, 85])

# 定义预测函数
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

# 统计预测正确率
theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if (a == 1 and b == 1) or (a == 0 and b == 0) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
# accuracy = 89%
print('accuracy = {0}%'.format(accuracy))

# 2 正则化逻辑回归
# 2.1 数据可视化
path2 = 'ex2_data/ex2data2.txt';
data_init = pd.read_csv(path2, header=None, names=['Test 1', 'Test 2', 'Accepted'])
positive2 = data_init[data_init['Accepted'].isin([1])]
negative2 = data_init[data_init['Accepted'].isin([0])]
'''
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(positive2['Test 1'], positive2['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative2['Test 1'], negative2['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()
'''

# 2.2 特征映射
# 一种更好的使用数据集的方式是为每组数据创造更多的特征
# 所以我们为每组x1 x2添加了最高到6次幂的特征
degree = 6
data2 = data_init
x1 = data2['Test 1']
x2 = data2['Test 2']

data2.insert(3, 'Ones', 1)

for i in range(1, degree + 1):
    for j in range(0, i + 1):
        data2['F' + str(i - j) + str(j)] = np.power(x1, i - j) * np.power(x2, j)

# .drop：Dataframe删除指定行列
# axis：axis=0表示行，axis=1表示列
data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)

# 2.3 代价函数+梯度下降
# 实现正则化的代价函数
def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg

# 实现正则化的梯度函数
def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:, i])
    return grad

cols = data2.shape[1]
X2 = data2.iloc[:, 1:cols]
y2 = data2.iloc[:, 0:1]
theta2 = np.zeros(cols - 1)
# 进行类型转换
X2 = np.array(X2.values)
y2 = np.array(y2.values)
# λ设为1
learningRate = 1
costReg(theta2, X2, y2, learningRate)

# 2.3.1 用工具库求解参数
result2 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))
theta_min = np.matrix(result2[0])
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))
# accuracy = 98%
print('accuracy = {0}%'.format(accuracy))

# 2.4 绘制决策边界
def hfunc2(theta, x1, x2):
    temp = theta[0][0]
    place = 0
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            temp += np.power(x1, i - j) * np.power(x2, j) * theta[0][place + 1]
            place += 1
    return temp

def find_decision_boundary(theta):
    t1 = np.linspace(-1, 1.5, 1000)
    t2 = np.linspace(-1, 1.5, 1000)

    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    h_val = pd.DataFrame({'x1': x_cord, 'x2': y_cord})
    h_val['hval'] = hfunc2(theta, h_val['x1'], h_val['x2'])

    decision = h_val[np.abs(h_val['hval']) < 2 * 10 ** -3]
    return decision.x1, decision.x2

'''
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(positive2['Test 1'], positive2['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative2['Test 1'], negative2['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')

x, y = find_decision_boundary(result2)
plt.scatter(x, y, c='y', s=10, label='Prediction')
ax.legend()
plt.show()
'''

# 2.5 改变lamda 观察决策曲线
# lamda = 0 过拟合
'''
learningRate2 = 0
result3 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate2))
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(positive2['Test 1'], positive2['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative2['Test 1'], negative2['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
x, y = find_decision_boundary(result3)
plt.scatter(x, y, c='y', s=10, label='Prediction')
ax.legend()
plt.show()
'''
# lamda = 100 欠拟合
'''
learningRate3 = 100;
result4 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate3))
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(positive2['Test 1'], positive2['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative2['Test 1'], negative2['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
x, y = find_decision_boundary(result4)
plt.scatter(x, y, c='y', s=10, label='Prediction')
ax.legend()
plt.show()
'''