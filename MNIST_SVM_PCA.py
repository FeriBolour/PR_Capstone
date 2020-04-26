import numpy as np
from keras.datasets import mnist
from sklearn import svm
import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


(x_train, y_train), (x_test, y_test) = mnist.load_data()
images = x_test

x_train = x_train.reshape(x_train.shape[0], 28*28)

new_col = np.ones(x_train.shape[0]).reshape(x_train.shape[0],1)
x_train =  np.append(x_train,new_col,axis=1)

x_test = x_test.reshape(x_test.shape[0], 28*28)
new_col = np.ones(x_test.shape[0]).reshape(x_test.shape[0],1)
x_test =  np.append(x_test,new_col,axis=1)

sc = StandardScaler()
x_tr = sc.fit_transform(x_train)
x_te = sc.transform(x_test)

# minimum number of principal components such that 95% of the variance is retained
pca = PCA(0.59)
pca.fit(x_tr);
x_tr = pca.transform(x_tr)
x_te = pca.transform(x_te)

start_time = time.time()

clf = svm.SVC(C = 1.0,kernel = 'rbf')
clf.fit(x_tr,y_train)

y_pred = clf.predict(x_te)
    
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

print("--- %s seconds ---" % (time.time() - start_time))