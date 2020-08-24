from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression as LR
import scipy.io as sio

# 加载数据
mat_train = sio.loadmat('./data/spamTrain.mat')
# print(mat_tr.keys())
X = mat_train['X']
y = mat_train['y']
# print(X.shape, y.shape)  # (4000, 1899) (4000, 1)

mat_test = sio.loadmat('./data/spamTest.mat')
# print(mat_test.keys())
X_test = mat_test['Xtest']
y_test = mat_test['ytest']
# print(X_test.shape, y_test.shape)  # (1000, 1899) (1000, 1)

# fit svm model
svc = svm.SVC()
svc = svc.fit(X, y.ravel())
# print(svc)  # C=1.0 gamma='auto'
y_pred = svc.predict(X_test)
print(metrics.classification_report(y_test,y_pred))  # total F1-score: 0.95

# fit logisticregression model
# lr = LR()
# lr = lr.fit(X, y.ravel())
# # print(lr)
# y_pred = lr.predict(X_test)
# print(metrics.classification_report(y_test, y_pred))  # total F1-score: 0.99
