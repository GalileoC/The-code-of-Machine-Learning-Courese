# -*-encoding:utf-8-*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./data/ex1data1.txt',names = ['population','profit']) #读取数据并赋予列名
# sns.lmplot('population','profit',df,fit_reg=False)

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
    return df.apply(lambda column: (column - column.mean()) / column.std(ddof=0)) #特征缩放

X = get_X(df)
y = get_y(df)

theta = np.zeros(X.shape[1])#X.shape[1]=2,代表特征数n #参数theta初始化为0

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
#批量梯度下降函数

epoch = 10
final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch)

# visualize cost data（代价数据可视化）
# ax = sns.tsplot(cost_data, time=np.arange(epoch+1))
# ax.set_xlabel('epoch')
# ax.set_ylabel('cost')
# plt.show()
#可以看到从第二轮代价数据变换很大，接下来平稳了

b = final_theta[0] # intercept，Y轴上的截距
m = final_theta[1] # slope，斜率

plt.scatter(df.population, df.profit, label="Training data")
plt.plot(df.population, df.population*m + b, label="Prediction")
plt.legend(loc=2)
plt.show()