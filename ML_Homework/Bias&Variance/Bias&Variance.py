import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
data_path = './data/ex5data1.mat'
def load_data(path):
    data = sio.loadmat(path)
    X = data['X']
    y = data['y']
    X_test = data['Xtest']
    y_test = data['ytest']
    X_val = data['Xval']
    y_val = data['yval']

    return X, y, X_test, y_test, X_val, y_val

X, y, X_test, y_test, X_val, y_val = load_data(data_path)
# print(X.shape)  # (12, 1)
# print(y.shape)  # (12, 1)

# 可视化
# plt.figure()
# plt.scatter(X, y, s=20)
# plt.show()

# 插入偏置值b
# X, X_val, X_test = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in [X, X_val, X_test]]

# cost function
def cost(theta, X, y):
    m = X.shape[0]

    inner = (X @ theta) - y.flatten()  # @：矩阵乘法运算
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)
    return cost

# 初始化
# theta = np.ones(X.shape[1])

# cost = cost(theta, X, y)
# print(cost)  # 303.9515255535976

# 计算梯度
def gradient(theta, X, y):
    m = X.shape[0]
    inner = X.T @ ((X @ theta) - y.flatten())
    return inner / m

# grad = gradient(theta, X, y)

# 正则化梯度
def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = theta.copy()
    regularized_term[0] = 0
    regularized_term = (l / m) * regularized_term
    return gradient(theta, X, y) + regularized_term

# re_grad = regularized_gradient(theta, X, y)

# 拟合数据
def regularized_cost(theta, X, y, l=1):
    m = X.shape[0]

    regularized_term = (l / (2 * m)) * np.power(theta[1:], 2).sum()

    return cost(theta, X, y) + regularized_term

def linear_regression_np(X, y, l=1):
    # init theta
    theta = np.ones(X.shape[1])

    # train it
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': True})
    return res

# final_theta = linear_regression_np(X, y, l=0).get('x')
# b = final_theta[0] # intercept
# m = final_theta[1] # slope

# plt.scatter(X[:,1], y, label="Training data")
# plt.plot(X[:, 1], X[:, 1]*m + b, label="Prediction")
# plt.legend(loc=2)
# plt.show()

# 训练误差和校验误差（训练集大小）
# training_cost, cv_cost = [], []
# m = X.shape[0]
# for i in range(1, m + 1):
#     #     print('i={}'.format(i))
#     res = linear_regression_np(X[:i, :], y[:i], l=0)
#
#     tc = regularized_cost(res.x, X[:i, :], y[:i], l=0)
#     cv = regularized_cost(res.x, X_val, y_val, l=0)
#     #     print('tc={}, cv={}'.format(tc, cv))
#
#     training_cost.append(tc)
#     cv_cost.append(cv)
#
# plt.plot(np.arange(1, m+1), training_cost, label='training cost')
# plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
# plt.legend(loc=1)
# plt.show()

# 使用多项式回归
# 创建多项式特征
def poly_features(x, power, as_ndarray=False):
    data = {'f{}'.format(i): np.power(x, i).flatten() for i in range(1, power + 1)}
    df = pd.DataFrame(data)

    return df.values if as_ndarray else df

def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())

def prepare_poly_data(*args, power):
    def prepare(x):
        # expand feature
        df = poly_features(x, power=power)

        # normalization
        ndarr = normalize_feature(df).values

        # add intercept term
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)

    return [prepare(x) for x in args]

X_poly, X_val_poly, X_test_poly = prepare_poly_data(X, X_val, X_test, power=8)

# 绘制数据集大小的学习曲线
def plot_learning_curve(X, y, X_val, y_val, l=0):
    training_cost, cv_cost = [], []
    m = X.shape[0]

    for i in range(1, m + 1):
        # regularization applies here for fitting parameters
        res = linear_regression_np(X[:i, :], y[:i], l=l)

        # remember, when you compute the cost here, you are computing
        # non-regularized cost. Regularization is used to fit parameters only
        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, X_val, y_val)

        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1, m + 1), training_cost, label='training cost')
    plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    plt.legend(loc=1)

# plot_learning_curve(X_poly, y, X_val_poly, y_val, l=100)
# plt.show()

# 使用学习曲线优化l
l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost = [], []
for l in l_candidate:
    res = linear_regression_np(X_poly, y, l)

    tc = cost(res.x, X_poly, y)
    cv = cost(res.x, X_val_poly, y_val)

    training_cost.append(tc)
    cv_cost.append(cv)

plt.plot(l_candidate, training_cost, label='training')
plt.plot(l_candidate, cv_cost, label='cross validation')
plt.xticks(l_candidate)
plt.legend(loc=2)
plt.xlabel('lambda')
plt.ylabel('cost')
plt.show()











