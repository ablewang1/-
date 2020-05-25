# 线性支持向量机
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from sklearn import metrics


x,y = datasets.make_blobs(n_features=2,n_samples=100,centers=2,cluster_std=2,random_state=20)
x = x.astype(np.float32)
y = y * 2 - 1
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.train(x_train,cv2.ml.ROW_SAMPLE,y_train)


_, y_pred = svm.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred))


def plot_decision_boundary(svm,x_test,y_test):
    x_min,x_max = x_test[:,0].min() -1 , x_test[:,0].max() +1
    y_min,y_max = x_test[:,1].min() -1 , x_test[:,1].max() +1
    h = 0.02
    xx,yy =np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

    x_hypo = np.c_[xx.ravel().astype(np.float32),yy.ravel().astype(np.float32)]
    # x_prod = np.c_[xx.flatten().astype(np.float32), yy.flatten().astype(np.float32)]

    _,zz =svm.predict(x_hypo)
    zz = zz.reshape(xx.shape)
    plt.contourf(xx,yy,zz,cmap=plt.cm.coolwarm,alpha=0.8)

    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=200)


plot_decision_boundary(svm,x_test,y_test)


plt.scatter(x[:, 0], x[:, 1], c=y, s=200)

plt.show()