import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import nltk
import re

from sklearn.datasets import fetch_openml
dataset = fetch_openml('mnist_784')

#this dataset contains 70k images of handwritten digits .the images have a 
#dimension of 28*28.images are flattened. to view an actual image we will
# have to reshape it back to 28*28 dimensions.

X = dataset.data
y = dataset.target
y = y.astype('int32')

#X,Y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y,test_size = 0.3, random_state = 0)

#print(X[0]) # 
#HSV = hue saturation value ---->0-255
def print_digit(index):
    some_digit = X[index]
    some_digit_image = some_digit.reshape(28,28)
    
    plt.imshow(some_digit_image,"binary")
    plt.show()
print_digit( 6999)

X_plot = X.reshape((70000, 28 ,28))
for i in range (25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_plot[i],"binary")
    plt.axis('off')
    plt.xlabel('label : {}'.format(y[i]))
plt.show()

#print_multiple_digits()

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

from sklearn.svm import SVC
svm = SVC()

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()

log_reg.fit(X_train, y_train)
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)
nb.fit(X_train, y_train)
dtf.fit(X_train, y_train)

log_reg.score(X_test,y_test)
knn.score(X_test,y_test)
svm.score(X_test,y_test)
nb.score(X_test,y_test)
dtf.score(X_test,y_test)














