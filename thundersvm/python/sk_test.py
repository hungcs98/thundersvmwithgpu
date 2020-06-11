from thundersvm import *
from sklearn.datasets import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing
import pandas as pd


balance_data=pd.read_csv("../dataset/car.data", sep= ',', header= None)

le = preprocessing.LabelEncoder()
balance_data = balance_data.apply(le.fit_transform)
# X chứa data của feature
x = balance_data.values[:, 0:5]
# y là target
y = balance_data.values[:,6]

# Chia data thành 2 phần: training data, test data
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2, random_state = 100)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
# Convert target dạng chữ về số
clf_gini.fit(x_train, y_train)


x,y = load_svmlight_file("../dataset/car.data")
clf = svc(verbose=true, gamma=0.5, c=100)
clf.fit(x,y)

x2,y2=load_svmlight_file("../dataset/car.data")

y_predict=clf.predict(x2)
score=clf.score(x2,y2)
clf.save_to_file('./model')

print ("test score is ", score)




