import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # seaborn: statistical data visualization
plt.style.use('fivethirtyeight')  # plt.style，主要用来选择plot的呈现风格。
from sklearn.metrics import classification_report  # 这个包是评价报告，用于显示主要分类指标的文本报告。在报告中显示每个类的精确度，召回率，F1值等信息。
import scipy.optimize as opt

# 1.读取数据
data = pd.read_csv('./data/ex2data1.txt', names=['exam1','exam2','admitted'])
# print(data.head()) # 查看前五行数据。
# print(data.describe()) # 查看数据信息，std为标准差。

# 2.数据可视化
# sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2),color_codes=False)
# sns.lmplot('exam1', 'exam2', hue='admitted', data=data
#            ,size=6
#            ,fit_reg=False
#            ,scatter_kws={"s": 50}
#           )
# plt.show()
# X = data.values
# X1 = X[:,0]
# X2 = X[:,1]
# y = X[:,2]
# plt.scatter(X1,X2,c=y,s=40,cmap=plt.cm.Spectral) # c:色彩或颜色序列；s:标量，表示点的大小；cmap:Colormap。
# plt.show()

# 3.划分数据
def get_X(df): # 读取特征
    """
    use concat to add intersect feature to avoid side effect
    not efficient for big dataset though
    """
    ones = pd.DataFrame({'ones': np.ones(len(df))})  # ones是m行1列的dataframe，此项为0次项。
    data_temp = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并。
    return data_temp.iloc[:, :-1].values  # 这个操作返回 ndarray,不是矩阵。
"""
iloc和loc：
无论是iloc还是loc 均采用[]而不是括号。
如果只是取行 建议用iloc 因为比较简单。
如果列和行同时取 建议采用loc 因为可以直接定义到标签。
"""

def get_y(df):  # 读取标签
    """
    assume the last column is the target
    """
    return np.array(df.iloc[:, -1])  # df.iloc[:, -1]是指df的最后一列

def normalize_feature(df):
    """
    Applies function along input axis(default 0) of DataFrame.
    Z-Score规范化：
    将原数据转换为正态分布的形式，使结果易于比较。
    公式为：新数值 = （原数值 - 均值）/ 标准差。
    """
    return df.apply(lambda column: (column - column.mean()) / column.std())  # 特征缩放

X = get_X(data)
y = get_y(data)

# 3.计算代价函数
def sigmoid(z):
    return 1 / (1+np.exp(-z))
# sigmoid函数图像
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(np.arange(-10, 10, step=0.01)
#         ,sigmoid(np.arange(-10, 10, step=0.01)))
# ax.set_ylim((-0.1,1.1))
# ax.set_xlabel('z', fontsize=18)
# ax.set_ylabel('g(z)', fontsize=18)
# ax.set_title('sigmoid function', fontsize=18)
# plt.show()

theta = np.zeros(3)

def cost(theta, X, y):
    ''' cost fn is -l(theta) for you to minimize'''
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))

# X @ theta与X.dot(theta)等价

# 4.梯度项计算
def gradient(theta, X, y):
    '''just 1 batch gradient'''
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)

# 5.拟合参数
res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='Newton-CG', jac=gradient)

# 6.用训练集预测和验证
def predict(x, theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)  # astype进行强制类型转换。

final_theta = res.x
y_pred = predict(X, final_theta)
# print(classification_report(y, y_pred))

# 7.寻找决策边界
coef = -(res.x / res.x[2])  # find the equation # 这里将y项的系数转换为1，方便后面画图。
# coef = final_theta

x1 = np.arange(130, step=0.1)
y1 = coef[0] + coef[1]*x1
# y1 = (coef[0] + coef[1]*x1) / (-coef[2])

sns.set(context="notebook", style="ticks", font_scale=1.5)

sns.lmplot('exam1', 'exam2', hue='admitted', data=data,
           size=6,
           fit_reg=False,
           scatter_kws={"s": 25}
          )

plt.plot(x1, y1, 'grey')
plt.xlim(0, 130)
plt.ylim(0, 130)
plt.title('Decision Boundary')
plt.show()