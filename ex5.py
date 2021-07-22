# 偏差和方差，训练集&验证集&测试集
import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# 1 正则化线性回归
# 1.1 数据可视化
data = sio.loadmat('ex5_data/ex5data1.mat')
# map() 会根据提供的函数对指定序列做映射
# np.ravel：将多维数组降为一维
# ((12,), (12,), (21,), (21,), (21,), (21,))
X,y,Xval,yval,Xtest,ytest = map(np.ravel,[data['X'],data['y'],data['Xval'],data['yval'],data['Xtest'],data['ytest']])
# 散点图
'''
fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(X,y)
ax.set_xlabel('water_level')
ax.set_ylabel('flow')
plt.show()
'''

# 1.2 正则化线性回归代价函数
X, Xval, Xtest = [np.insert(x.reshape(x.shape[0],1),0,np.ones(x.shape[0]),axis=1) for x in (X,Xval,Xtest)]
# 定义代价函数
def cost(theta,X,y):
    # 维度
    m = X.shape[0]
    # h_theta(x^(i))
    inner = X @ theta - y
    square_num = inner.T @ inner

    cost = square_num / (2 * m)
    return cost
def costReg(theta,X,y,reg=1):
    m = X.shape[0]
    # lamda/2m   (theta_j ^(2))
    regularized_term = (reg / (2 * m)) * np.power(theta[1:],2).sum()
    return cost(theta,X,y) + regularized_term

theta = np.ones(X.shape[1])
# 303.9931922202643
costReg(theta,X,y,1)

# 1.3 正则化线性回归的梯度
def gradient(theta, X,y):
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)
    return inner / m

def gradientReg(theta,X,y,reg):
    m = X.shape[0]
    #  copy() 函数返回一个字典的浅复制
    regularized_term = theta.copy()
    regularized_term[0] = 0
    regularized_term = (reg / m) * regularized_term
    return gradient(theta,X,y) + regularized_term

# [-15.30301567 598.25074417]
gradientReg(theta,X,y,1)

# 1.4 拟合线性回归
# numpy.ones（）函数返回给定形状和数据类型的新数组，其中元素的值设置为1
theta = np.ones(X.shape[1])
# 调用工具库
final_theta = opt.minimize(fun=costReg,x0=theta,args=(X,y,0),method='TNC',jac=gradientReg,options={'disp':True}).x
print(final_theta)

b = final_theta[0]
m = final_theta[1]
# 绘制预测结果
'''
fig,ax = plt.subplots(figsize=(8,4))
plt.scatter(X[:,1],y,c='r',label="Training data")
plt.plot(X[:,1],X[:,1]*m+b,c='b', label="Prediction")
ax.set_xlabel('water_level')
ax.set_ylabel('flow')
ax.legend()
plt.show()
'''

# 2 方差和偏差
# 机器学习中的一个重要概念是偏差-方差权衡。
# 偏差较大的模型会欠拟合，而方差较大的模型会过拟合。
# 这部分会让你画出学习曲线来判断方差和偏差的问题。
# 2.1 学习曲线
# 使用训练集的子集来拟合应模型
# 在计算训练代价和验证集代价时，没有正则化
# 记住使用相同的训练子集来计算训练代价
def linear_regression(X,y,l=1):
    theta = np.ones(X.shape[1])
    res = opt.minimize(fun=costReg,x0=theta,args=(X,y,l),method='TNC',jac=gradientReg,options={'disp':True})
    return res

training_cost,cv_cost = [],[]
m = X.shape[0]
for i in range(1, m+1):
    res = linear_regression(X[:i,:],y[:i],0)
    tc = costReg(res.x,X[:i,:],y[:i],0)
    cv = costReg(res.x,Xval,yval,0)
    training_cost.append(tc)
    cv_cost.append(cv)

# 欠拟合
'''
fig,ax = plt.subplots(figsize=(8,4))
plt.plot(np.arange(1,m+1),training_cost,label="training cost")
plt.plot(np.arange(1,m+1),cv_cost,label="cv cost")
plt.legend()
plt.show()
'''

# 3 多项式回归
# 线性回归对于现有数据来说太简单了，会欠拟合，我们需要多添加一些特征。
# 写一个函数，输入原始X，和幂的次数p，返回X的1到p次幂
def ploy_features(x, power, as_ndarray=False):
    data = {'f{}'.format(i):np.power(x,i) for i in range(1, power+1)}
    df = pd.DataFrame(data)
    return df.values if as_ndarray else df

data = sio.loadmat('ex5_data/ex5data1.mat')
X,y,Xval,yval,Xtest,ytest = map(np.ravel,[data['X'],data['y'],data['Xval'],data['yval'],data['Xtest'],data['ytest']])
ploy_features(X,power=3)

# 3.1 多项式回归
# 使用之前的代价函数和梯度函数
# 扩展特征到8阶特征
# 使用 归一化 来处理 x^(n)
# λ = 0
def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())

def prepare_poly_data(*args, power):
    def prepare(x):
        df = ploy_features(x, power=power)

        ndarr = normalize_feature(df).values

        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]),axis=1)
    return [prepare(x) for x in args]

X_poly,Xval_poly,Xtest_poly = prepare_poly_data(X,Xval,Xtest,power=8)

def plot_learning_curve(X,Xinit,y,Xval,yval,l=0):
    training_cost,cv_cost = [],[]

    m = X.shape[0]

    for i in range(1,m+1):
        res = linear_regression(X[:i,:], y[:i],l=1)

        tc = cost(res.x, X[:i,:], y[:i])
        cv = cost(res.x, Xval, yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    fig,ax = plt.subplots(2, 1, figsize=(8,8))
    ax[0].plot(np.arange(1, m+1), training_cost, label='training cost')
    ax[0].plot(np.arange(1, m+1), cv_cost, label='cv cost')
    ax[0].legend()

    fitx = np.linspace(-50, 50, 100)
    fitxtmp = prepare_poly_data(fitx, power=8)
    fity = np.dot(prepare_poly_data(fitx, power=8)[0], linear_regression(X,y,l).x.T)

    ax[1].plot(fitx, fity, c='r', label="fitcurve")
    ax[1].scatter(Xinit, y, c='b', label="initial_Xy")

    ax[1].set_xlabel('water_level')
    ax[1].set_ylabel('flow')

# 过拟合
# plot_learning_curve(X_poly,X,y,Xval_poly,yval,l=0)
# plt.show()

# 3.2 调整正则化系数λ
# λ = 1 减轻了过拟合
# plot_learning_curve(X_poly,X,y,Xval_poly,yval,l=1)
# plt.show()

# λ = 100 欠拟合
# plot_learning_curve(X_poly, X, y, Xval_poly, yval, l=100)
# plt.show()

# 3.3 找到最佳的λ
l_candidate = [0, 0.001, 0.003, 0.01, 0.03 ,0.1, 0.3, 1, 3, 10]
training_cost,cv_cost = [],[]

for l in l_candidate:
    res = linear_regression(X_poly,y,l)
    tc = cost(res.x,X_poly,y)
    cv = cost(res.x,Xval_poly,yval)
    training_cost.append(tc)
    cv_cost.append(cv)

'''
fig,ax = plt.subplots(figsize=(8,4))
ax.plot(l_candidate, training_cost, label='training')
ax.plot(l_candidate, cv_cost, label='cross validation')

plt.legend()
plt.xlabel('lambda')
plt.ylabel('cost')
plt.show()
'''

# 3.4 计算测试集上的误差
for l in l_candidate:
    theta = linear_regression(X_poly,y,l).x
    print('test cost(l={})={}'.format(l,cost(theta,Xtest_poly,ytest)))
