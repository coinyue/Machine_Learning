# 反向传播神经网络
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

# 1 神经网络
# 再次处理手写数字数据集，使用反向传播的前馈神经网络；
# 1.1 数据可视化
data = loadmat('ex4_data/ex4data1.mat')
# (5000, 400)
X = data['X']
# (5000, 1)
y = data['y']
weight = loadmat('ex4_data/ex4weights.mat')
# (25, 401), (10, 26)
theta1, theta2 = weight['Theta1'], weight['Theta2']

sample_idx = np.random.choice(np.arange(data['X'].shape[0]), 100)
sample_images = data['X'][sample_idx,:]
fig, ax_array = plt.subplots(nrows=10,ncols=10,sharey=True,sharex=True,figsize=(8,8))
for r in range(10):
    for c in range(10):
        ax_array[r,c].matshow(np.array(sample_images[10*r+c].reshape((20,20))).T, cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
# plt.show()

# 1.2 模型展示
# 与ex3相同

# 1.3 前向传播和代价函数
def sigmoid(z):
    return 1 / (1+np.exp(-z))

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)

    return a1,z2,a2,z3,h

# 代价函数
def cost(theta1, theta2, input_size, hidden_size, num_labels, X, y,learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    a1,z2,a2,z3,h = forward_propagate(X, theta1, theta2)

    J = 0;
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1-y[i,:]), np.log(1-h[i,:]))
        J += np.sum(first_term - second_term)
    J = J / m
    return J

encoder = OneHotEncoder(sparse=False)
# (5000, 10)
y_onehot = encoder.fit_transform(y)

# 初始化设置
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1
# 0.2876291651613187
result = cost(theta1, theta2, input_size, hidden_size, num_labels,X,y_onehot,learning_rate)

# 1.4 正则化代价函数
def costReg(theta1, theta2, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    a1,z2,a2,z3,h = forward_propagate(X, theta1, theta2)
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1-y[i,:]), np.log(1-h[i,:]))
        J += np.sum(first_term - second_term)
    J = J / m
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:],2)) + np.sum(np.power(theta2[:,1:],2)))
    return J

# 0.3837698590909234
costReg(theta1, theta2, input_size, hidden_size, num_labels,X,y_onehot,learning_rate)


# 2 反向传播
# 2.1 sigmoid梯度
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1-sigmoid(z)))

# 0.25
sigmoid_gradient(0)

# 2.2 随机初始化
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.24

# 2.3 反向传播
def backprop(params, input_size, hidden_size, num_labels, X,y,learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # 正向传播
    a1,z2,a2,z3,h = forward_propagate(X, theta1, theta2)

    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size,(input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # 初始化
    J = 0
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)

    # 计算代价
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1-y[i,:]), np.log(1-h[i,:]))
        J += np.sum(first_term - second_term)

    J = J / m

    for t in range(m):
        alt = a1[t,:]
        z2t = z2[t,:]
        a2t = a2[t,:]
        ht = h[t,:]
        yt = y[t,:]

        d3t = ht - yt

        z2t = np.insert(z2t, 0, values=np.ones(1))
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))

        delta1 = delta1 + (d2t[:,1:]).T * alt
        delta2 = delta2 + d3t.T + a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    return J, delta1, delta2

# 2.4 梯度校验

# 2.5 正则化神经网络 加入正则项
def backpropReg(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    a1,z2,a2,z3,h = forward_propagate(X, theta1, theta2)

    J = 0
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)

    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1-y[i,:]), np.log(1-h[i,:]))

        J += np.sum(first_term - second_term)

    J = J / m

    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:],2)) + np.sum(np.power(theta2[:,1:],2)))

    for t in range(m):
        a1t = a1[t,:]
        z2t = z2[t,:]
        a2t = a2[t,:]
        ht = h[t,:]
        yt = y[t,:]

        d3t = ht - yt

        z2t = np.insert(z2t, 0, values=np.ones(1))
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))

        delta1 = delta1 + (d2t[:,1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m

    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad

# 2.6 使用工具库计算参数最优解
fmin = minimize(fun=backpropReg, x0 = (params), args=(input_size,hidden_size,num_labels,X,y_onehot,learning_rate),method='TNC',jac=True,options={'maxiter':250})
print(fmin)
X = np.matrix(X)
thetafinal1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
thetafinal2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):],(num_labels,(hidden_size + 1))))
a1,z2,a2,z3,h = forward_propagate(X, thetafinal1, thetafinal2)
y_pred = np.array(np.argmax(h, axis=1)+1)
print(y_pred)
print(classification_report(y, y_pred))

# 3 可视化隐藏层
hidden_layer = thetafinal1[:,1:]
print(hidden_layer.shape)
'''
fig,ax_array = plt.subplots(nrows=5,ncols=5,sharey=True, sharex=True,figsize=(8,8))
for r in range(5):
    for c in range(5):
        ax_array[r,c].matshow(np.array(hidden_layer[5*r + c].reshape((20,20))),cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()
'''