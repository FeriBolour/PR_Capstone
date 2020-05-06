import numpy as np
import os
import glob
import cv2
from sklearn import svm
import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


x = []
y = []

#directory = input("Please enter your storage point")
start_time = time.time()

img_dir = r"V:\celegans\0\training" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)
    if img.shape == (101,101):
        x.append(np.array([img, 0]))


img_dir = r"V:\celegans\1\training" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)
    if img.shape == (101,101):
        x.append(np.array([img, 1]))

img_dir = r"V:\celegans\0\test" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)
    if img.shape == (101,101):
        y.append(np.array([img, 0]))


img_dir = r"V:\celegans\1\test" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)
    if img.shape == (101,101):
        y.append(np.array([img, 1]))

x = np.array(x)
y = np.array(y)


np.random.seed(1)  
np.random.shuffle(x)
np.random.shuffle(y)

x_train = x[:, 0]
y_train = x[:, 1:2]
x_test = y[:, 0]
y_test = y[:, 1:2]

train_size = y_train.shape[0]
test_size = y_test.shape[0]
    
x__train = np.zeros([train_size, 101,101])
x__test = np.zeros([test_size, 101,101])


for i in range(len(x_train)):
    x__train[i] = x_train[i]
    
for i in range(len(x_test)):
    x__test[i] = x_test[i]
    
    
x_train = x__train
x_test = x__test

mean = 0
std = 0
x_train = x_train.reshape(x_train.shape[0], 101*101)
mean = np.mean(x_train)
std = np.std(x_train)
x_train = (x_train-mean)/std
new_col = np.ones(train_size).reshape(train_size,1)
x_train =  np.append(x_train,new_col,axis=1)

x_test = x_test.reshape(x_test.shape[0], 101*101)
x_test = (x_test-mean)/std
new_col = np.ones(test_size).reshape(test_size,1)
x_test =  np.append(x_test,new_col,axis=1)
    
y_train = np.array(y_train,dtype= 'f')
y_test = np.array(y_test,dtype= 'f')


sc = StandardScaler()
x_tr = sc.fit_transform(x_train)
x_te = sc.transform(x_test)


# minimum number of principal components such that 95% of the variance is retained
pca = PCA(0.95)
pca.fit(x_tr);
x_tr = pca.transform(x_tr)
x_te = pca.transform(x_te)

clf = svm.SVC(C = 1.0,kernel = 'rbf')

start_time = time.time()
clf.fit(x_tr,y_train)
print("--- %s seconds for training---" % (time.time() - start_time))

start_time = time.time()
y_pred = clf.predict(x_te)
print("--- %s seconds for testing---" % (time.time() - start_time))
    
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))



