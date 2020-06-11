import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing
import pandas as pd

from thundersvm import *
from sklearn.datasets import *

balance_data=pd.read_csv("../dataset/car.data",
                           sep= ',', header= None)
#print "Dataset:: "

#df1.head()

le = preprocessing.LabelEncoder()
balance_data = balance_data.apply(le.fit_transform)
# X chứa data của feature
X = balance_data.values[:, 0:5]
# y là target
Y = balance_data.values[:,6]

# Chia data thành 2 phần: training data, test data
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 100)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
# Convert target dạng chữ về số
clf_gini.fit(X_train, y_train)
print("",balance_data)


clf = SVC(verbose=True, gamma=0.5, C=100)
clf.fit(X,Y)

y_predict=clf.predict(X)
score=clf.score(X,Y)
clf.save_to_file('./model')
print ("test score is ", score)

Y_decision=clf.decision_function(X)
print("decision_function",Y_decision)
