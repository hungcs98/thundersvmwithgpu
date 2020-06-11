from svm import *
y,x = svm_read_problem('../dataset/test_dataset.txt')
svm_train(y,x,'test_dataset.txt.model','-c 100 -g 0.5')
y,x=svm_read_problem('../dataset/test_dataset.txt')
svm_predict(y,x,'test_dataset.txt.model','test_dataset.predict')