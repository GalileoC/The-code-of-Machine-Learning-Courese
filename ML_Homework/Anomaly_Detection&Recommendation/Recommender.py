import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="white", palette=sns.color_palette("RdBu"), color_codes=False)
import numpy as np
import pandas as pd
import scipy.io as sio

# Load data and setting up
# Notes: X - num_movies (1682)  x num_features (10) matrix of movie features
#        Theta - num_users (943)  x num_features (10) matrix of user features
#        Y - num_movies x num_users matrix of user ratings of movies
#        R - num_movies x num_users matrix, where R(i, j) = 1 if the i-th movie was rated by the j-th user

data = sio.loadmat('./data/ex8_movies.mat')
Y, R = data['Y'], data['R']
# print(Y.shape, R.shape)  # (1682, 943) (1682, 943)

m, u = Y.shape  # movies and users
n = 10  # features of a movie

param = sio.loadmat('./data/ex8_movieParams.mat')
theta, X = param.get('Theta'), param.get('X')
# print(theta.shape, X.shape)  # (943, 10) (1682, 10)
# print(param['num_users'], param['num_movies'], param['num_features'])

# Cost
def serialize(X, theta):
    """serialize 2 matrix
    """
    # X (movie, feature), (1682, 10): movie features
    # theta (user, feature), (943, 10): user preference
    return np.concatenate((X.ravel(), theta.ravel()))


def deserialize(param, n_movie, n_user, n_features):
    """into ndarray of X(1682, 10), theta(943, 10)"""
    return param[:n_movie * n_features].reshape(n_movie, n_features), \
           param[n_movie * n_features:].reshape(n_user, n_features)


# recommendation fn
def cost(param, Y, R, n_features):
    """compute cost for every r(i, j)=1
    Args:
        param: serialized X, theta
        Y (movie, user), (1682, 943): (movie, user) rating
        R (movie, user), (1682, 943): (movie, user) has rating
    """
    # theta (user, feature), (943, 10): user preference
    # X (movie, feature), (1682, 10): movie features
    n_movie, n_user = Y.shape
    X, theta = deserialize(param, n_movie, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)

    return np.power(inner, 2).sum() / 2

def gradient(param, Y, R, n_features):
    # theta (user, feature), (943, 10): user preference
    # X (movie, feature), (1682, 10): movie features
    n_movies, n_user = Y.shape
    X, theta = deserialize(param, n_movies, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)  # (1682, 943)

    # X_grad (1682, 10)
    X_grad = inner @ theta

    # theta_grad (943, 10)
    theta_grad = inner.T @ X

    # roll them together and return
    return serialize(X_grad, theta_grad)

def regularized_cost(param, Y, R, n_features, l=1):
    reg_term = np.power(param, 2).sum() * (l / 2)

    return cost(param, Y, R, n_features) + reg_term

def regularized_gradient(param, Y, R, n_features, l=1):
    grad = gradient(param, Y, R, n_features)
    reg_term = l * param

    return grad + reg_term

# use subset of data to calculate the cost as in pdf...
# users = 4
# movies = 5
# features = 3
#
# X_sub = X[:movies, :features]
# theta_sub = theta[:users, :features]
# Y_sub = Y[:movies, :users]
# R_sub = R[:movies, :users]
#
# param_sub = serialize(X_sub, theta_sub)
# print(cost(param_sub, Y_sub, R_sub, features))

# Gradient
# n_movie, n_user = Y.shape
#
# X_grad, theta_grad = deserialize(gradient(param, Y, R, 10),
#                                       n_movie, n_user, 10)
#
# assert X_grad.shape == X.shape
# assert theta_grad.shape == theta.shape

# Regularized cost
# in the ex8_confi.m, lambda = 1.5, and it's using sub data set
# regularized_cost(param_sub, Y_sub, R_sub, features, l=1.5)
#
# regularized_cost(param, Y, R, 10, l=1)  # total regularized cost

# Regularized gradient
# n_movie, n_user = Y.shape
#
# X_grad, theta_grad = deserialize(regularized_gradient(param, Y, R, 10),
#                                                                 n_movie, n_user, 10)
#
# assert X_grad.shape == X.shape
# assert theta_grad.shape == theta.shape

# Parse movie_id.txt
movie_list = []

with open('./data/movie_ids.txt', encoding='latin-1') as f:
    for line in f:
        tokens = line.strip().split(' ')
        movie_list.append(' '.join(tokens[1:]))

movie_list = np.array(movie_list)

# Reproduce my ratings
ratings = np.zeros(1682)

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

# Prepare data
Y, R = data.get('Y'), data.get('R')


Y = np.insert(Y, 0, ratings, axis=1)  # now I become user 0
print(Y.shape)

R = np.insert(R, 0, ratings != 0, axis=1)
print(R.shape)

n_features = 50
n_movie, n_user = Y.shape
l = 10

X = np.random.standard_normal((n_movie, n_features))
theta = np.random.standard_normal((n_user, n_features))

print(X.shape, theta.shape)

param = serialize(X, theta)

Y_norm = Y - Y.mean()
print(Y_norm.mean())

# Training
import scipy.optimize as opt
res = opt.minimize(fun=regularized_cost,
                   x0=param,
                   args=(Y_norm, R, n_features, l),
                   method='TNC',
                   jac=regularized_gradient)

print(res)

X_trained, theta_trained = deserialize(res.x, n_movie, n_user, n_features)
print(X_trained.shape, theta_trained.shape)

prediction = X_trained @ theta_trained.T

my_preds = prediction[:, 0] + Y.mean()

idx = np.argsort(my_preds)[::-1]  # Descending order
print(idx.shape)

# top ten idx
print(my_preds[idx][:10])

for m in movie_list[idx][:10]:
    print(m)

