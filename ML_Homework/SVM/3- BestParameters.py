# 使用网格搜索寻找最佳参数

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics  # 评估指标
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

data = sio.loadmat('./data/ex6data3.mat')
X = data['X']
y = data['y']
X_val = data['Xval']
y_val = data['yval']
# print(X.shape)  # (211, 2)
# print(y.shape)  # (211, 1)

# plt.figure()
# plt.scatter(X[:,0], X[:,1], c=y.ravel(), s=10, cmap=plt.cm.Spectral)
# plt.show()

# 人工优化参数
candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]  # C和gamma的取值范围
# combination = [(C, gamma) for C in candidate for gamma in candidate]
#
# search = []
# for C, gamma in combination:
#     svc = svm.SVC(C=C, gamma=gamma)
#     svc.fit(X, y.ravel())
#     search.append(svc.score(X_val, y_val))
#
# best_score = search[np.argmax(search)]  # 取出元素最大值所对应的索引（索引值默认从0开始）
# best_param = combination[np.argmax(search)]
#
# # print(best_score, best_param)  # 0.965 (0.3, 100)
#
# best_svc = svm.SVC(C=0.3, gamma=100)
# best_svc = best_svc.fit(X, y.ravel())
# ypred = best_svc.predict(X_val)
# print(metrics.classification_report(y_val, ypred))

# 网格搜索
parameters = {'C':candidate,'gamma':candidate}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, n_jobs=-1)  # the computation will be dispatched on all the CPUs of the computer
clf.fit(X, y.ravel())

# print(clf.best_params_)  # {'C': 10, 'gamma': 30}
# print(clf.best_score_)  # 0.9004739336492891
pred = clf.predict(X_val)
print(metrics.classification_report(y_val, pred))

# 两次结果不同的原因：
# 错误原因：人工优化是在校验集中寻找最佳参数，而网格搜索是在训练集中寻找最佳参数。
# 正确原因：GridSearch会将部分数据作为CV并使用它来查找最佳候选者。因此产生不同结果的原因只是GridSearch此处只是使用部分训练数据进行训练，因为它需要将部分数据作为cv set。
