import numpy as np
import pandas as pd
# from scipy.io import loadmat  # 读取mat文件。
import scipy.io as sio  # 读取mat文件。
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report#这个包是评价报告

datapath = './data/ex3data1.mat'
# data = loadmat(datapath)  # type为dict。
# X = data['X']  # type为np.ndarray, shape为(5000,400)
# y = data['y']  # type为np.ndarray, shape为(5000,1)

def load_data(path, transpose=True):
    data = sio.loadmat(path)
    y = data.get('y')  # (5000,1)
    y = y.reshape(y.shape[0])  # make it back to column vector（将标签转换为一位数组）

    X = data.get('X')  # (5000,400)

    if transpose:  # 将图片方向转正。
        # for this dataset, you need a transpose to get the orientation right
        X = np.array([im.reshape((20, 20)).T for im in X])  # 将一个样本数据转换为二维数组并进行转置。

        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])  # 将二维数组转换为一维数组。

    return X, y

# raw_X, raw_y = load_data(datapath)

# 绘图函数
def plot_an_image(image):
    """
    image : (400,)  # 输入图片数据为一维数组。
    """
    # plt.subplots()是一个函数，返回一个包含figure和axes对象的元组。因此，使用fig,ax = plt.subplots()将元组分解为fig和ax两个变量。
    fig, ax = plt.subplots(figsize=(1, 1))

    # matshow()进行矩阵可视化。
    ax.matshow(image.reshape(20, 20), cmap=matplotlib.cm.binary)

    # 去掉刻度。
    plt.xticks(np.array([]))  # just get rid of ticks
    plt.yticks(np.array([]))

# pick_one = np.random.randint(0, 5000)
# plot_an_image(X[pick_one, :])
# plt.show()
# print('this should be {}'.format(y[pick_one]))

# 绘图函数，画100张图片
def plot_100_image(X):
    """ sample 100 image and show them
    assume the image is square

    X : (5000, 400)
    """
    size = int(np.sqrt(X.shape[1]))

    # sample 100 image, reshape, reorg it
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 100*400
    sample_images = X[sample_idx, :]

    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))
    # sharex和sharey表示坐标轴的属性是否相同，可选的参数：True，False，row，col，默认值均为False，表示画布中的四个ax是相互独立的。

    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(sample_images[10 * r + c].reshape((size, size))
                                   ,cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
# plot_100_image(X)
# plt.show()

# theta = sio.loadmat('./data/ex3weights.mat')
# Theta1 = theta.get('Theta1')
# Theta2 = theta.get('Theta2')

# 准备数据
# add intercept=1 for x0  # 插入x0
# X = np.insert(raw_X, 0, values=np.ones(raw_X.shape[0]), axis=1)  # 插入了第一列（全部为1）
# print(X.shape)  # (5000, 401)

# 对y进行编码。
# y have 10 categories here. 1..10, they represent digit 0 as category 10 because matlab index start at 1
# I'll ditit 0, index 0 again
# y_matrix = []

# for k in range(1, 11):
#     y_matrix.append((raw_y == k).astype(int))    # 见配图 "向量化标签.png"

# last one is k==10, it's digit 0, bring it to the first position，最后一列k=10，都是0，把最后一列放到第一列
# y_matrix = [y_matrix[-1]] + y_matrix[:-1]
# y = np.array(y_matrix)

# 扩展 5000*1 到 5000*10
#     比如 y=10 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]: ndarray
#     """
# print(y.shape) # (10, 5000)

# train 1 model (训练一维模型)
# 代价函数
def cost(theta, X, y):
    ''' cost fn is -l(theta) for you to minimize'''
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))

# 正则项函数
def regularized_cost(theta, X, y, l=1):
    '''you don't penalize theta_0'''
    theta_j1_to_n = theta[1:]
    regularized_term = (l / (2 * len(X))) * np.power(theta_j1_to_n, 2).sum()

    return cost(theta, X, y) + regularized_term


def regularized_gradient(theta, X, y, l=1):
    '''still, leave theta_0 alone'''
    theta_j1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_n

    # by doing this, no offset is on theta_0
    regularized_term = np.concatenate([np.array([0]), regularized_theta])

    return gradient(theta, X, y) + regularized_term

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient(theta, X, y):
    '''just 1 batch gradient'''
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)

def logistic_regression(X, y, l=1):
    """generalized logistic regression
    args:
        X: feature matrix, (m, n+1) # with incercept x0=1
        y: target vector, (m, )
        l: lambda constant for regularization

    return: trained parameters
    """
    # init theta
    theta = np.zeros(X.shape[1])

    # train it
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': True})
    # get trained parameters
    final_theta = res.x

    return final_theta

def predict(x, theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)

# t0 = logistic_regression(X, y[0])

# print(t0.shape)
# y_pred = predict(X, t0)
# print('Accuracy={}'.format(np.mean(y[0] == y_pred)))

# train k model (训练k维模型)
# # 对y的每一个标签都训练得到theta参数
# k_theta = np.array([logistic_regression(X, y[k]) for k in range(10)])
# print(k_theta.shape)
#
# # 预测值矩阵。
# prob_matrix = sigmoid(X @ k_theta.T)
# np.set_printoptions(suppress=True)
# # print(prob_matrix)
# y_pred = np.argmax(prob_matrix, axis=1)  # 返回沿轴axis最大值的索引，axis=1代表行
# # print(y_pred)
#
# # 得到结果。
# y_answer = raw_y.copy()
# y_answer[y_answer==10] = 0
# print(classification_report(y_answer, y_pred))

# 神经网络模型。
X, y = load_data('./data/ex3data1.mat',transpose=False)

X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)  # intercept

# 加载theta参数
def load_weight(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']

theta1, theta2 = load_weight('./data/ex3weights.mat')

# print(theta1.shape, theta2.shape)  # (25, 401) (10, 26)

# X, y = load_data('ex3data1.mat',transpose=False)
#
# X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)  # intercept
#
# X.shape, y.shape  # ((5000, 401), (5000,))

# feed forward prediction（前馈预测）
# 计算第一层隐藏层。
a1 = X
z2 = a1 @ theta1.T # (5000, 401) @ (25,401).T = (5000, 25)
# print(z2.shape)  # (5000, 25)

# 插入偏置值。
z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)
a2 = sigmoid(z2)
# print(a2.shape)  # (5000, 26)。

# 计算第二层隐藏层。
z3 = a2 @ theta2.T
# print(z3.shape)  # (5000, 10)
a3 = sigmoid(z3)

# 输出前向传播的预测值。
y_pred = np.argmax(a3, axis=1) + 1  # numpy is 0 base index, +1 for matlab convention，返回沿轴axis最大值的索引，axis=1代表行
# print(y_pred.shape)  # (5000,)

# 得到准确率
print(classification_report(y, y_pred))

