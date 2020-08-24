import numpy as np
import pandas as pd
import scipy.optimize as opt
import seaborn as sns
import matplotlib.pyplot as plt

# 1.读取数据
df = pd.read_csv('./data/ex2data2.txt',names=['test1','test2','accepted'])
# 2.数据可视化
# sns.set(context="notebook", style="ticks", font_scale=1.5)
#
# sns.lmplot('test1', 'test2', hue='accepted', data=df,
#            size=6,
#            fit_reg=False,
#            scatter_kws={"s": 50}
#           )
#
# plt.title('Regularized Logistic Regression')
# plt.show()
# X1 = df['test1'].values
# X2 = df['test2'].values
# y = df['accepted'].values
# plt.figure()
# plt.scatter(X1,X2,c=y,s=40,cmap=plt.cm.Spectral)
# plt.show()

# 获取数据
def get_y(df):
    return np.array(df.iloc[:,-1])

# 3.特征映射，当逻辑回归问题较复杂，原始特征不足以支持构建模型时，可以通过组合原始特征成为多项式，创建更多特征，使得决策边界呈现高阶函数的形状，从而适应复杂的分类问题。
def feature_mapping(x, y, power, as_ndarray=False):
    """return mapped features as ndarray or dataframe"""
    # data = {}
    # # inclusive
    # for i in np.arange(power + 1):
    #     for p in np.arange(i + 1):
    #         data["f{}{}".format(i - p, p)] = np.power(x, i - p) * np.power(y, p)

    data = {"f{}{}".format(i - p, p): np.power(x, i - p) * np.power(y, p)
                for i in np.arange(power + 1)
                for p in np.arange(i + 1)
            }

    if as_ndarray:
        return pd.DataFrame(data).as_matrix()
    else:
        return pd.DataFrame(data)

x1 = np.array(df.test1)
x2 = np.array(df.test2)

data = feature_mapping(x1,x2,power=6)

# 4.计算正则化函数
theta = np.zeros(data.shape[1])
X = feature_mapping(x1, x2, power=6, as_ndarray=True)

y = get_y(df)

# 计算代价函数
def sigmoid(z):
    return 1 / (1+np.exp(-z))
def cost(theta, X, y):
    ''' cost fn is -l(theta) for you to minimize'''
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))

# 正则化代价函数
def regularized_cost(theta, X, y, l=1):
    '''you don't penalize theta_0'''
    theta_j1_to_n = theta[1:]
    regularized_term = (l / (2 * len(X))) * np.power(theta_j1_to_n, 2).sum()

    return cost(theta, X, y) + regularized_term

# 梯度项计算
def gradient(theta, X, y):
    '''just 1 batch gradient'''
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)

# 正则化梯度
def regularized_gradient(theta, X, y, l=1):
    '''still, leave theta_0 alone'''
    theta_j1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_n

    # by doing this, no offset is on theta_0
    regularized_term = np.concatenate([np.array([0]), regularized_theta])

    return gradient(theta, X, y) + regularized_term

# 5.拟合参数
print('init cost = {}'.format(regularized_cost(theta, X, y)))

res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y), method='Newton-CG', jac=regularized_gradient)

# 6.预测数据
def predict(x, theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)

final_theta = res.x
y_pred = predict(X, final_theta)

# 7.画出决策边界
def feature_mapped_logistic_regression(power, l):
    """for drawing purpose only.. not a well generealize logistic regression
    power: int
        raise x1, x2 to polynomial power
    l: int
        lambda constant for regularization term
    """
    df = pd.read_csv('./data/ex2data2.txt', names=['test1', 'test2', 'accepted'])
    x1 = np.array(df.test1)
    x2 = np.array(df.test2)
    y = get_y(df)

    X = feature_mapping(x1, x2, power, as_ndarray=True)
    theta = np.zeros(X.shape[1])

    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient)
    final_theta = res.x
    return final_theta

# 寻找决策边界函数
def find_decision_boundary(density, power, theta, threshhold):
    t1 = np.linspace(-1, 1.5, density)
    t2 = np.linspace(-1, 1.5, density)

    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    mapped_cord = feature_mapping(x_cord, y_cord, power)  # this is a dataframe

    inner_product = mapped_cord.as_matrix() @ theta

    decision = mapped_cord[np.abs(inner_product) < threshhold]

    return decision.f10, decision.f01



def draw_boundary(power, l):
    """
    power: polynomial power for mapped feature
    l: lambda constant
    """
    density = 1000
    threshhold = 2 * 10**-3

    final_theta = feature_mapped_logistic_regression(power, l)
    x, y = find_decision_boundary(density, power, final_theta, threshhold)

    df = pd.read_csv('./data/ex2data2.txt', names=['test1', 'test2', 'accepted'])
    sns.lmplot('test1', 'test2', hue='accepted', data=df, size=6, fit_reg=False, scatter_kws={"s": 100})

    plt.scatter(x, y, c='R', s=10)
    plt.title('Decision boundary')
    plt.show()

draw_boundary(power=6, l=1)  # lambda=1z