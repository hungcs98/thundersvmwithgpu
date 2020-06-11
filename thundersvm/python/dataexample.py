from thundersvm import *
from sklearn.datasets import *

x,y = load_svmlight_file("../dataset/test_dataset.txt")
clf = SVC(verbose=True, gamma=0.5, C=100)
clf.fit(x,y)

x2,y2=load_svmlight_file("../dataset/test_dataset.txt")
y_predict=clf.predict(x2)

score=clf.score(x2,y2)
clf.save_to_file('./model')

print ("test score is ", score)