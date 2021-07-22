# 异常检测和推荐系统（协同过滤）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from scipy import stats
from scipy.optimize import minimize

# 1 异常检测
# 使用高斯模型来检测数据集中未标记的示例是否应被视为异常
# 二维数据
data = loadmat('ex8_data/ex8data1.mat')
X = data['X']
# 数据可视化
'''
fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(X[:,0], X[:,1])
plt.show()
'''

# 1.1 高斯分布
# 1.2 计算高斯分布参数
def estimate_gaussian(X):
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)

    return mu, sigma

mu, sigma = estimate_gaussian(X)
# 数据可视化
'''
xplot = np.linspace(0,25,100)
yplot = np.linspace(0,25,100)
# np.meshgrid：生成网格点坐标矩阵
Xplot, Yplot = np.meshgrid(xplot, yplot)
Z = np.exp((-0.5) * ((Xplot - mu[0]) ** 2 / sigma[0] + (Yplot - mu[1]) ** 2) / sigma[1])
fig,ax = plt.subplots(figsize=(8,4))
contour = plt.contour(Xplot, Yplot, Z, [10**-11, 10**-7, 10**-3, 0.1], colors='k')
ax.scatter(X[:,0],X[:,1])
plt.show()
'''

# 1.3 选择阈值ε
# 有了参数后，可以估计每组数据的概率
# 低概率的数据点更可能是异常的。确定异常点需要先确定一个阈值
# 我们可以通过验证集集来确定这个阈值。
# ((307, 2), (307, 1))
Xval = data['Xval']
yval = data['yval']
# scipy.stats.norm函数 可以实现正态分布（也就是高斯分布）
# 计算数据点属于正态分布的概率
dist = stats.norm(mu[0], sigma[0])
p = np.zeros((X.shape[0], X.shape[1]))
# 计算并保存给定上述的高斯模型参数的数据集中每个值的概率密度
p[:,0] = stats.norm(mu[0], sigma[0]).pdf(X[:,0])
p[:,1] = stats.norm(mu[1], sigma[1]).pdf(X[:,1])
# 为验证集（使用相同的模型参数）执行此操作
# 我们将使用与真实标签组合的这些概率来确定将数据点分配为异常的最佳概率阈值
pval = np.zeros((Xval.shape[0], Xval.shape[1]))
pval[:,0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:,0])
pval[:,1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:,1])

# 1.4 找到给定概率密度值和真实标签的最佳阈值
def select_threshold(pval, yval):
    best_epsilon = 0
    best_f1 = 0
    # f1 = 0

    step = (pval.max() - pval.min()) / 1000

    for epsilon in np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon

        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1

epsilon, f1 = select_threshold(pval, yval)
print(epsilon, f1)

outliers = np.where(p < epsilon)
print(outliers)

fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(X[:,0], X[:,1])
ax.scatter(X[outliers[0],0], X[outliers[0],1],s=50,color='r',marker='o')
plt.show()


# 协同过滤
data = loadmat('ex8_data/ex8_movies.mat')
Y = data['Y']
R = data['R']
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(Y)
ax.set_xlabel('Users')
ax.set_ylabel('Movies')
fig.tight_layout()
plt.show()

def serialize(X, theta):
    return np.concatenate((X.ravel(), theta.ravel()))

def deserialize(param, n_movie, n_user, n_features):
    return param[:n_movie * n_features].reshape(n_movie, n_features), param[n_movie * n_features:].reshape(n_user, n_features)


def cost(param, Y, R, n_features):
    n_movie, n_user = Y.shape
    X, theta = deserialize(param, n_movie, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)

    return np.power(inner,2).sum() / 2

params_data = loadmat('C:/Users/DELL/Desktop/ex8_movieParams.mat')
X = params_data['X']
theta = params_data['Theta']

users = 4
movies = 5
features = 3
X_sub = X[:movies, :features]
theta_sub = theta[:users, :features]
Y_sub = Y[:movies, :users]
R_sub = R[:movies, :users]

param_sub = serialize(X_sub, theta_sub)

re = cost(param_sub,Y_sub, R_sub,features)
print(re)

param = serialize(X, theta)
res = cost(serialize(X, theta), Y, R, 10)
print(res)

# 梯度下降
def gradient(param, Y, R, n_features):
    n_movies, n_user = Y.shape
    X, theta = deserialize(param, n_movies, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)

    X_grad = inner @ theta

    theta_grad = inner.T @ X

    return serialize(X_grad, theta_grad)
n_movies, n_user = Y.shape
X_grad, theta_grad = deserialize(gradient(param, Y, R,10), n_movies, n_user,10)

def regularized_cost(param, Y, R, n_features, l=1):
    reg_term = np.power(param, 2).sum() * (l/2)

    return cost(param, Y, R, n_features) + reg_term

def regularized_gradient(param, Y, R, n_features, l=1):
    grad = gradient(param, Y, R,n_features)
    reg_term = l * param

    return grad + reg_term
res1 = regularized_cost(param_sub, Y_sub, R_sub, features, l=1.5)
res2 = regularized_cost(param, Y,R, 10, l=1)
n_movie, n_user = Y.shape
X_grad, theta_grad = deserialize(regularized_gradient(param, Y, R, 10), n_movie, n_user,10)
movie_list = []
f = open('C:/Users/DELL/Desktop/movie_ids.txt', "r",encoding='ISO-8859-1')

for line in f:
    tokens = line.strip().split(' ')
    movie_list.append(' '.join(tokens[1:]))

movie_list = np.array(movie_list)
print(movie_list[0])
ratings = np.zeros((1682, 1))

ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5


Y = data['Y']
Y = np.append(ratings,Y, axis=1)  
print(Y.shape)


R = data['R']
R = np.append( ratings != 0, R,axis=1)

movies = Y.shape[0] 
users = Y.shape[1]  
features = 10
learning_rate = 10.

X = np.random.random(size=(movies, features))
theta = np.random.random(size=(users, features))
params = serialize(X, theta)

Y_norm = Y - Y.mean()
print(Y_norm.mean())

fmin = minimize(fun=regularized_cost, x0=params, args=(Y_norm, R, features, learning_rate), 
                method='TNC', jac=regularized_gradient)


X_trained, theta_trained = deserialize(fmin.x, movies, users, features)

prediction = X_trained @ theta_trained.T
my_preds = prediction[:, 0] + Y.mean()
idx = np.argsort(my_preds)[::-1]

my_preds[idx][:10]


for m in movie_list[idx][:10]:
    print(m)