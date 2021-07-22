# 线性回归
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 2.1 读入数据 展示数据
path = 'ex1_data/ex1data1.txt'
data = pd.read_csv(path, header = None, names=['Population', 'Profit'])
# print(data.head())
# data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
# plt.show()
# 2.2 梯度下降
# 在现有数据集上 训练线性回归参数theta

# 2.2.1 J(theta)
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

# 这个是为了绘图的
# 计算代价函数
def computeCost1(X,y,theta):
    m = len(y)
    result = np.dot(X , theta)
    result = result - y.reshape(97,1)
    result = np.square(result)
    result = np.sum( result,axis=0)
    result = result/(2.0*float(m))
    return result
# 2.2.2实现
# 数据前面已经读取完毕，我们要为加入一列x，用于更新
# 然后我们将初始化为0，学习率初始化为0.01，迭代次数为1500次
data.insert(0, 'Ones', 1)
# 初始化X和y
cols = data.shape[1]
# iloc函数：通过行号来取行数据
# X是data里的除最后列
X = data.iloc[:,:-1]
# y是data最后一列
y = data.iloc[:,cols-1:cols]
# 代价函数是应该是numpy矩阵，所以我们需要转换X和Y，然后才能使用它们。
# 我们还需要初始化theta
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

# 2.2.3 计算J(theta)
resultCost = computeCost(X,y,theta)
print(resultCost)

# 2.2.4 梯度下降 minJ(theta)
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

alpha = 0.01
iters = 1500
# return theta, cost
g, cost = gradientDescent(X, y, theta, alpha, iters)
predict1 = [1,3.5]*g.T
print("predict1:",predict1)
predict2 = [1,7]*g.T
print("predict2:",predict2)

# np.linspace：在指定的间隔内返回均匀间隔的数字
x = np.linspace(data.Population.min(), data.Population.max(), 100)
# f = theta0 + theta1*x
f = g[0, 0] + (g[0, 1] * x)

# 原始数据以及拟合的直线可视化
# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(x, f, 'r', label='Prediction')
# ax.scatter(data.Population, data.Profit, label='Traning Data')
# ax.legend(loc=2)
# ax.set_xlabel('Population')
# ax.set_ylabel('Profit')
# ax.set_title('Predicted Profit vs. Population Size')
# plt.show()

# 2.4 可视化J(theta)
'''
# 绘制三维的图像
fig = plt.figure()
axes3d = Axes3D(fig)
# 指定参数的区间
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
# 存储代价函数值的变量初始化
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
# 为代价函数的变量赋值
for i in range(0,len(theta0_vals)):
    for j in range(0,len(theta1_vals)):
        t = np.zeros((2,1))
        t[0] = theta0_vals[i]
        t[1] = theta1_vals[j]
        J_vals[i,j]  = computeCost1(X, y, t)
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals) #必须加上这段代码
axes3d.plot_surface(theta0_vals,theta1_vals,J_vals, rstride=1, cstride=1, cmap='rainbow')
plt.show()
'''

# 3 多变量线性回归
path2 = 'ex1_data/ex1data2.txt'
data2 = pd.read_csv(path2, header=None, names=['Size', 'Bedrooms', 'Price'])

# 3.1 特征归一化
data2 = (data2 - data2.mean()) / data2.std()

# 3.2 梯度下降
# 加一列常数项
data2.insert(0, 'Ones', 1)
# 初始化X和y
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]
# 转换成matrix格式，初始化theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))
alpha = 0.01
iters = 1500
# 运行梯度下降算法
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)
print(g2)

# 3.3 正规方程（区别于迭代方法的直接解法）
# 正规方程
def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y#X.T@X等价于X.T.dot(X)
    return theta

final_theta2=normalEqn(X, y)#这里用的是data1的数据
print(final_theta2)