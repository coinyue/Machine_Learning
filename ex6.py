# 支持向量机
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm

# 1 支持向量
# 1.1 数据集1-线性边界 2D数据集
raw_data = loadmat('ex6_data/ex6data1.mat')
# data 第一个参数是存放在DataFrame里的数据
# index 第二个参数index就是之前说的行名
# columns 第三个参数columns是之前说的列名
data = pd.DataFrame(raw_data.get('X'), columns=['X1','X2'])
data['y'] = raw_data.get('y')
'''
          X1      X2  y
0   1.964300  4.5957  1
1   2.275300  3.8589  1
2   2.978100  4.5651  1
3   2.932000  3.5519  1
4   3.577200  2.8560  1
5   4.015000  3.1937  1
'''

# 1.1.2 数据可视化
def plot_init_data(data, fig, ax):
    positive = data[data['y'].isin([1])]
    negative = data[data['y'].isin([0])]

    ax.scatter(positive['X1'], positive['X2'], s = 50, marker='x', label='Positive')
    ax.scatter(negative['X1'], negative['X2'], s = 50, marker='o', label='Negative')

'''
fig, ax = plt.subplots(figsize=(8,4))
plot_init_data(data, fig, ax)
ax.legend()
plt.show()
'''
# 1.1.2 SVC
# 令 C=1（C：正则化参数）
svc = svm.LinearSVC(C = 1, loss='hinge', max_iter=1000)
# svc.fit：根据给定数据训练模型
svc.fit(data[['X1','X2']], data['y'])
# svc.score：返回给定测试数据和标签的平均准确性
# 0.9803921568627451
svc.score(data[['X1','X2']],data['y'])

# 1.1.3 可视化分类边界
def find_decision_boundary(svc, x1min, x1max, x2min, x2max, diff):
    # 作为序列生成器， numpy.linspace()函数用于在线性空间中以均匀步长生成数字序列。
    # (1000,)
    x1 = np.linspace(x1min, x1max, 1000)
    # (1000,)
    x2 = np.linspace(x2min, x2max, 1000)

    # (x,y)
    cordinates = [(x, y) for x in x1 for y in x2]
    # zip() 函数用于将可迭代的对象作为参数
    # 将对象中对应的元素打包成一个个元组
    # 然后返回由这些元组组成的列表
    # ???
    x_cord, y_cord = zip(*cordinates)

    c_val = pd.DataFrame({'x1':x_cord, 'x2':y_cord})
    c_val['cval'] = svc.decision_function(c_val[['x1','x2']])
    decision = c_val[np.abs(c_val['cval']) < diff]
    return decision.x1, decision.x2

# 可视化
'''
x1,x2 = find_decision_boundary(svc,0,4,1.5,5,2*10**-3)
fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(x1,x2,s=10,c='r',label="Boundary")
plot_init_data(data,fig,ax)
ax.set_title('SVM (C=1) Decision Boundary')
ax.legend()
plt.show()
'''

# 令C=100（之前C=1）
svc2 = svm.LinearSVC(C = 100, loss='hinge', max_iter=1000)
svc2.fit(data[['X1','X2']], data['y'])
# 0.9019607843137255
svc2.score(data[['X1','X2']],data['y'])
'''
x1,x2 = find_decision_boundary(svc2,0,4,1.5,5,2*10**-3)
fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(x1,x2,s=10,c='r',label="Boundary")
plot_init_data(data,fig,ax)
ax.set_title('SVM (C=100) Decision Boundary')
ax.legend()
plt.show()
'''

# 1.2 高斯内核的SVM
# 现在我们将从线性SVM转移到能够使用内核进行非线性分类的SVM
# 1.2.1 高斯内核
# 你把高斯内核认为是一个衡量一对数据间的“距离”的函数
# 有一个参数，决定了相似性下降至0有多快
def gaussian_kernel(x1, x2, sigma):
    return np.exp(-(np.sum((x1-x2)**2) / (2*(sigma**2))))

x1 = np.array([1.0,2.0,1.0])
x2 = np.array([0.0,4.0,-1.0])
sigma = 2
# 0.32465246735834974
gaussian_kernel(x1, x2, sigma)

# 1.2.2 数据集2-非线性边界
raw_data = loadmat('ex6_data/ex6data2.mat')
data = pd.DataFrame(raw_data['X'], columns=['X1','X2'])
data['y'] = raw_data['y']
# 数据可视化
'''
fig,ax = plt.subplots(figsize=(8,4))
plot_init_data(data, fig, ax)
ax.legend()
plt.show()
'''
# 训练支持向量机
# C=100
svc = svm.SVC(C=100,gamma=10,probability=True)
svc.fit(data[['X1','X2']], data['y'])
# 0.9698725376593279
svc.score(data[['X1','X2']], data['y'])
# 边界可视化
'''
x1 ,x2 = find_decision_boundary(svc, 0, 1, 0.4, 1, 0.01)
fig,ax = plt.subplots(figsize=(8,4))
plot_init_data(data, fig, ax)
ax.scatter(x1, x2, s=10)
plt.show()
'''

# 1.2.3 数据集3-给出训练和验证集
# 并且基于验证集性能为SVM模型找到最优超参数
raw_data = loadmat('ex6_data/ex6data3.mat')
X = raw_data['X']
Xval = raw_data['Xval']
y = raw_data['y'].ravel()
yval = raw_data['yval'].ravel()
# 数据可视化
'''
fig,ax = plt.subplots(figsize=(8,4))
data = pd.DataFrame(raw_data.get('X'),columns=['X1','X2'])
data['y'] = raw_data.get('y')
plot_init_data(data, fig, ax)
plt.show()
'''
# 尝试不同的超参数
C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

best_score = 0
best_params = {'C':None, 'gamma':None}
for C in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=C, gamma=gamma)
        svc.fit(X,y)
        score = svc.score(Xval,yval)
        # 寻找得分最高的参数
        if score > best_score:
            best_score = score
            best_params['C'] = C
            best_params['gamma'] = gamma
best_score, best_params

# 以找到的最好参数来训练SVM
svc = svm.SVC(C = best_params['C'], gamma=best_params['gamma'])
svc.fit(X,y)
'''
x1, x2 = find_decision_boundary(svc, -0.6, 0.3, -0.7, 0.6, 0.005)
fig,ax = plt.subplots(figsize=(8,4))
plot_init_data(data, fig, ax)
ax.scatter(x1, x2, s=10)
plt.show()
'''

# 2 垃圾邮件分类
# 使用SVM来构建垃圾邮件过滤器
# 2.1 处理邮件
# 2.2 提取特征
# 2.3 训练垃圾邮件分类SVM
spam_train = loadmat('ex6_data/spamTrain.mat')
spam_test = loadmat('ex6_data/spamTest.mat')
# (4000, 1899), (4000,), (1000, 1899), (1000,)
X = spam_train['X']
Xtest = spam_test['Xtest']
y = spam_train['y'].ravel()
ytest = spam_test['ytest'].ravel()
# 这个svm是刚才以最好的参数训出来的svm
svc = svm.SVC()
svc.fit(X, y)
print('Trainng accuracy = {0}%'.format(np.round(svc.score(X, y) * 100, 2)))
print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2)))

# 2.4 可视化结果
kw = np.eye(1899)
kw[:3,:]
spam_val = pd.DataFrame({'idx':range(1899)})
spam_val['isspam'] = svc.decision_function(kw)
re = spam_val['isspam'].describe()
decision = spam_val[spam_val['isspam'] > 0.55]
print(decision)
path = 'ex6_data/vocab.txt'
voc = pd.read_csv(path, header=None, names=['idx','voc'], sep='\t')
print(voc.head())
spamvoc = voc.loc[list(decision['idx'])]
print(spamvoc)