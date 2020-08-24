# -*-encoding:utf-8-*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

raw_data = pd.read_csv('./data/ex1data2.txt', names=['square','bedrooms','price'])

def get_X(df): #读取特征
    """
    use concat to add intersect feature to avoid side effect
    not efficient for big dataset though
    """
    ones = pd.DataFrame({'ones': np.ones(len(df))}) # ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并
    return data.iloc[:, :-1].values  # 这个操作返回 ndarray,不是矩阵

def get_y(df):#读取标签
    '''assume the last column is the target'''
    return np.array(df.iloc[:, -1])#df.iloc[:, -1]是指df的最后一列

def normalize_feature(df):
    """Applies function along input axis(default 0) of DataFrame."""
    return df.apply(lambda column: (column - column.mean()) / column.std()) #std()为标准差

data = normalize_feature(raw_data)
X = get_X(data)
y = get_y(data)

def lr_cost(theta, X, y):
    """
    X: R(m*n), m 样本数, n 特征数
    y: R(m)
    theta : R(n), 线性回归的参数
    """
    m = X.shape[0] # m为样本数

    inner = X @ theta - y  # R(m*1)，X @ theta等价于X.dot(theta)

    # 1*m @ m*1 = 1*1 in matrix multiplication
    # but you know numpy didn't do transpose in 1d array, so here is just a
    # vector inner product to itselves
    square_sum = inner.T @ inner # .T为求转置矩阵
    cost = square_sum / (2 * m)

    return cost

def gradient(theta, X, y):
    m = X.shape[0] # m为样本数

    inner = X.T @ (X @ theta - y)  # (m,n).T @ (m, 1) -> (n, 1)，X @ theta等价于X.dot(theta)

    return inner / m

def batch_gradient_decent(theta, X, y, epoch, alpha=0.01):
    """
       拟合线性回归，返回参数和代价
        epoch: 批处理的轮数
    """
    cost_data = [lr_cost(theta, X, y)]
    _theta = theta.copy()  # 拷贝一份，不和原来的theta混淆

    for _ in range(epoch):
        _theta = _theta - alpha * gradient(_theta, X, y)
        cost_data.append(lr_cost(_theta, X, y))

    return _theta, cost_data

alpha = 0.01#学习率
theta = np.zeros(X.shape[1])#X.shape[1]：特征数n
epoch = 500#轮数

# final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch, alpha=alpha)

# sns.tsplot(time=np.arange(len(cost_data)), data = cost_data)
# plt.xlabel('epoch', fontsize=18)
# plt.ylabel('cost', fontsize=18)
# plt.show()

# learning_rate
base = np.logspace(-1, -5, num=4)
candidate = np.sort(np.concatenate((base, base*3)))

# epoch=50
#
# fig, ax = plt.subplots(figsize=(16, 9))
#
# for alpha in candidate:
#     _, cost_data = batch_gradient_decent(theta, X, y, epoch, alpha=alpha)
#     ax.plot(np.arange(epoch+1), cost_data, label=alpha)
#
# ax.set_xlabel('epoch', fontsize=18)
# ax.set_ylabel('cost', fontsize=18)
# ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# ax.set_title('learning rate', fontsize=18)
# plt.show()

# 正规方程
def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y#X.T@X等价于X.T.dot(X)
    return theta

final_theta2=normalEqn(X, y)#感觉和批量梯度下降的theta的值有点差距

print(final_theta2)