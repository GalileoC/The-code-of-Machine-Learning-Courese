from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat('./data/ex4data1.mat')

X = data['X']
y = data['y']

def data_Visualization(image):
    plt.imshow(image.reshape(20,20).T, cmap='binary')
    plt.show()

data_Visualization(X[1234])
print(y[1234])
