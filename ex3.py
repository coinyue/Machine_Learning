# 多类别逻辑回归、前馈神经网络
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.metrics import classification_report

# 1 多分类
# 手写数字的识别 扩展逻辑回归 并将其应用于一对多的分类
# 1.1 读取数据
data = loadmat('ex3_data/ex3data1.mat')

# 1.2 数据可视化 随机展示100个数据
sample_idx = np.random.choice(np.arange(data['X'].shape[0]),100)
sample_images = data['X'][sample_idx,:]
fig, ax_array = plt.subplots(nrows=10,ncols=10,sharey=True,sharex=True,figsize=(12,12))
for r in range(10):
    for c in range(10):
        ax_array[r, c].matshow(np.array(sample_images[10*r+c].reshape((20,20))).T, cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
# plt.show()

# 1.3 将逻辑回归向量化 多分类
# 1.3.1 向量化代价函数
# sigmoid
def sigmoid(z):
    return 1/(1+np.exp(-z))

# 代价函数
def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X*theta.T)))
    second = np.multiply((1-y), np.log(1-sigmoid(X*theta.T)))
    reg = (learningRate / (2*len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first-second) / len(X) + reg

# 1.3.2 向量化梯度
# 1.3.3 向量化正则化逻辑回归
def gradient(theta, X, y,learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X*theta.T) - y
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)

    grad[0,0] = np.sum(np.multiply(error, X[:,0])) / len(X)

    return np.array(grad).ravel()

# 1.4 一对多分类器
def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]

    all_theta = np.zeros((num_labels, params+1))

    X = np.insert(X, 0, values=np.ones(rows),axis=1)

    for i in range(1, num_labels+1):
        theta = np.zeros(params+1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        fmin = minimize(fun=cost, x0 = theta, args=(X,y_i, learning_rate),method='TNC',jac=gradient)
        all_theta[i-1,:] = fmin.x
    return all_theta

rows = data['X'].shape[0]
params = data['X'].shape[1]
all_theta = np.zeros((10, params+1))
X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)
theta = np.zeros(params + 1)
y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = np.reshape(y_0, (rows, 1))

all_theta = one_vs_all(data['X'], data['y'], 10, 1)

# 1.4.1 一对多预测
def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    h = sigmoid(X*all_theta.T)

    h_argmax = np.argmax(h, axis=1)
    h_argmax = h_argmax + 1
    return h_argmax

y_pred = predict_all(data['X'], all_theta)
print(classification_report(data['y'], y_pred))

# 2 神经网络
# 2.1 模型表达
# 输入是图片的像素值，20*20像素的图片有400个输入层单元
# 不包括需要额外添加的加上常数项
# 材料已经提供了训练好的神经网络的参数theta1 theta2
# 有25个隐层单元和10个输出单元（10个输出）

# 2.2 前馈神经网络和预测
weight = loadmat('ex3_data/ex3weights.mat')
theta1, theta2 = weight['Theta1'], weight['Theta2']
# 插入常数项
# (5000, 401) add a0
X2 = np.matrix(np.insert(data['X'], 0, values=np.ones(X.shape[0]), axis=1))
# (5000, 1)
y2 = np.matrix(data['y'])

a1 = X2
z2 = a1 * theta1.T
a2 = sigmoid(z2)

# add a0
a2 = np.insert(a2, 0, values = np.ones(a2.shape[0]),axis=1)
z3 = a2 * theta2.T
a3 = sigmoid(z3)

y_pred2 = np.argmax(a3, axis=1) + 1
print(y_pred2.shape)
# sklearn中的classification_report函数用于显示主要分类指标的文本报告
# 在报告中显示每个类的精确度，召回率，F1值等信息
print(classification_report(y2, y_pred))