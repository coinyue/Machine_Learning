# K-means 和PCA（主成分分析）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from IPython.display import Image
from skimage import io
from sklearn.cluster import KMeans

# 1 K-means聚类
# 1.1 定义寻找数据i最近的中心函数
def find_closest_centroids(X, centroids):
    # 数据维度
    m = X.shape[0]
    # k个中心
    k = centroids.shape[0]
    # 用于记录中心
    idx = np.zeros(m)

    # 遍历所有数据
    for i in range(m):
        # 初始值
        min_dist = 1000000
        # 遍历k个中心
        for j in range(k):
            # 计算距离中心j的距离
            dist = np.sum((X[i,:] - centroids[j,:]) ** 2)
            # 寻找最小的
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    return idx

# 1.2 测试函数
data = loadmat('ex7_data/ex7data2.mat')
X = data['X']
initial_centroids = np.array([[3, 3],[6, 2], [8, 5]])
idx = find_closest_centroids(X, initial_centroids)

# 1.3 数据可视化
data2 = pd.DataFrame(data.get('X'), columns=['X1','X2'])
'''
sb.set(context="notebook", style="white")
sb.lmplot('X1','X2',data=data2,fit_reg=False)
plt.show()
'''

# 1.4 计算中心
def compute_centroids(X, idx, k):
    # X.shape (300, 2)
    m,n = X.shape
    # centroids (3, 2)
    centroids = np.zeros((k,n))
    # k个簇
    # idx[]：i距离最近的cluster标号
    for i in range(k):
        # np.where(condition)
        # 则输出满足条件元素的坐标
        # Indices：属于簇i的所有点
        indices = np.where(idx == i)
        # np.sum：axis为整数，axis的取值不可大于数组/矩阵的维度，且axis的不同取值会产生不同的结果
        # ravel()方法将数组维度拉成一维数组
        centroids[i,:] = (np.sum(X[indices,:],axis=1) / len(indices[0])).ravel()
    return centroids

compute_centroids(X, idx, 3)
'''
array([[2.42830111, 3.15792418],
       [5.81350331, 2.63365645],
       [7.11938687, 3.6166844 ]])
'''

# 1.5 k-means算法
def run_k_means(X, initial_centroids, max_iters):
    # # X.shape (300, 2)
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)
    return idx, centroids

# 计算3个簇中心
# 这里的initial_centroids：之前随便给的
idx, centroids  = run_k_means(X, initial_centroids, 10)
cluster1 = X[np.where(idx == 0)[0],:]
cluster2 = X[np.where(idx == 1)[0],:]
cluster3 = X[np.where(idx == 2)[0],:]

# 结果可视化
'''
fig,ax = plt.subplots(figsize=(8,4))
ax.scatter(cluster1[:,0], cluster1[:,1], s=30,color='r',label="Cluster 1")
ax.scatter(cluster2[:,0], cluster2[:,1], s=30,color='g',label="Cluster 2")
ax.scatter(cluster3[:,0], cluster3[:,1], s=30,color='b',label="Cluster 3")
ax.legend()
plt.show()
'''

# 1.6 随机初始化聚类中心
# 创建一个选择随机样本并将其用作初始聚类中心的函数
def init_centroids(X, k):
    m,n = X.shape
    centroids = np.zeros((k,n))
    idx = np.random.randint(0, m, k)

    for i in range(k):
        centroids[i,:] = X[idx[i],:]
    return centroids

res = init_centroids(X, 3)
print(res)

# 1.7 将K-means算法应用于图像压缩
# 可以使用聚类来找到最具代表性的少数颜色
# 并使用聚类分配将原始的24位颜色映射到较低维的颜色空间
Image(filename='ex7_data/bird_small.png')
image_data = loadmat('ex7_data/bird_small.mat')
# (128, 128, 3)
A = image_data['A']
# 归一化
A = A / 255
# X.shape：(16384, 3) 128*128=16384
X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
# 把图像当矩阵
# init_centroids(X, k) 16个cluster
initial_centroids = init_centroids(X, 16)
# run_k_means(X, initial_centroids, max_iters)
idx, centroids = run_k_means(X, initial_centroids, 10)
idx = find_closest_centroids(X, centroids)
# 压缩后图像可视化
'''
X_recovered = centroids[idx.astype(int),:]
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1],A.shape[2]))
plt.imshow(X_recovered)
plt.show()
'''

# 下面来用scikit-learn来实现K-means
pic = io.imread('ex7_data/bird_small.png') / 255
# 可视化
'''
io.imshow(pic)
plt.show()
'''
# data：(16384, 3)
data = pic.reshape(128 * 128, 3)
model = KMeans(n_clusters=16, n_init = 100, n_jobs = -1)
model.fit(data)

centroids = model.cluster_centers_
print(centroids.shape)
C = model.predict(data)
print(C.shape)

centroids[C].shape
compressed_pic = centroids[C].reshape((128, 128, 3))
# 可视化
'''
fig, ax = plt.subplots(1,2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)
plt.show()
'''

# 2 主成分分析
# 2.1 读取数据
data = loadmat('ex7_data/ex7data1.mat')
# (50, 2)
X = data['X']

# 2.2 数据可视化
'''
fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(X[:,0], X[:,1])
plt.show()
'''
# 2.3 pca函数
def pca(X):
    X = (X - X.mean()) / X.std()

    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]

    U,S,V = np.linalg.svd(cov)

    return U,S,V

# U：主成分
U,S,V = pca(X)

# 2.4 实现一个计算投影并且仅选择顶部K个分量的函数，有效地减少了维数
def project_data(X,U,k):
    U_reduced = U[:,:k]
    return np.dot(X, U_reduced)

Z = project_data(X,U,1)

# 2.5 反向转换步骤来恢复原始数据
def recover_data(Z,U,k):
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)

X_recovered = recover_data(Z,U,1)
# 可视化
'''
fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(list(X_recovered[:,0]), list(X_recovered[:,1]))
plt.show()
'''

# 2.6 将PCA应用于脸部图像。 通过使用相同的降维技术
# 我们可以使用比原始图像少得多的数据来捕获图像的“本质”
faces = loadmat('ex7_data/ex7faces.mat')
# X：(5000, 1024)
X = faces['X']

# 2.6.1 定义绘图函数
def plot_n_image(X, n):
    # 1024*1024
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))

    first_n_images = X[:n,:]

    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size, sharey=True, sharex=True,figsize=(8,8))

    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r,c].imshow(first_n_images[grid_size * r + c].reshape((pic_size,pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))

# 绘图
# plt.imshow(face)
# plt.show()
U,S,V = pca(X)
# 恢复
Z = project_data(X,U,100)
X_recovered = recover_data(Z,U,100)
face = np.reshape(X_recovered[3,:],(32,32))
'''
plt.imshow(face)
plt.show()
'''