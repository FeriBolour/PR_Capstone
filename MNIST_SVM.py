import numpy as np
from keras.datasets import mnist
from sklearn import svm
import time
from sklearn.metrics import classification_report, confusion_matrix

(x_train, y_train), (x_test, y_test) = mnist.load_data()
images = x_test

x_train = x_train.reshape(x_train.shape[0], 28*28)

new_col = np.ones(x_train.shape[0]).reshape(x_train.shape[0],1)
x_train =  np.append(x_train,new_col,axis=1)

x_test = x_test.reshape(x_test.shape[0], 28*28)
new_col = np.ones(x_test.shape[0]).reshape(x_test.shape[0],1)
x_test =  np.append(x_test,new_col,axis=1)

start_time = time.time()

clf = svm.SVC(C = 1.0,kernel = 'rbf')
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
    
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

print("--- %s seconds ---" % (time.time() - start_time))