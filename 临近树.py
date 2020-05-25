from sklearn import tree
import numpy as np
from matplotlib import pyplot as plt

rand1 = np.random.RandomState(32)
plt.style.use('ggplot')

x = np.sort(5*rand1.rand(100,1),axis=0)
y = np.sin(x).ravel()
y[::2] += 0.5*(0.5-rand1.rand(50))

reg1 = tree.DecisionTreeRegressor(max_depth=2,random_state=3)
reg2 = tree.DecisionTreeRegressor(max_depth=5,random_state=5)
reg1.fit(x,y)
reg2.fit(x,y)

x_test = np.arange(0.0,8.0,0.01)[:,np.newaxis]
y1 = reg1.predict(x_test)
y2 = reg2.predict(x_test)

plt.figure(figsize=(6,6))
plt.subplot(221)
plt.scatter(x,y,c="g",s=20,label="Data1")
plt.plot(x_test,y1,label="Deep_2",linewidth=2)
plt.subplot(222)
plt.scatter(x,y,c="b",s=20,label="Data2")
plt.plot(x_test,y2,label="Deep_5",linewidth=3)
plt.xlabel("DATA")
plt.ylabel("TARGET")
plt.show()