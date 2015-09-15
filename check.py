
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn import svm

w = np.array([[0, 1], [1, 2], [2, 1]])
a = sparse.lil_matrix((3, 3))
a[0, 0] = a[1, 1] = a[2, 2] = 1
x = a.dot(w)
y = np.arange(3)
svc = svm.SVC().fit(x, y)
x = np.r_[x, np.array([[3, svc.predict(np.array([[0, 0]]))]])]
plt.plot(x[:, 0], x[:, 1])
plt.show()
