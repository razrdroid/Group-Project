from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
#from sklearn.neighbors import KNeighboursClassifier
from sklearn.metrics import accuracy_score
import numpy as np
X= [[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],[175,64,39],
   [177,64,39],[159,70,43],[171,75,42],[181,85,43]]
   
Y= ['male','female','female','female','male','male','male',
    'male','female','male','female']
height = int(input())
weight = int(input())
size= int(input())
a=[[height,weight,size]]
clf_tree= tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
#clf_KNN = KneighboursClassifier()

clf_tree.fit(X,Y)
clf_svm.fit(X,Y)
clf_perceptron.fit(X,Y)
#clf_KNN.fit(X,Y)


pred_tree = clf_tree.predict(X)
acc_tree= accuracy_score(Y,pred_tree)*100
print('Accuracy for DecisionTreeClassifier:{}'.format(acc_tree))
prediction = clf_tree.predict(a)


print (prediction)

pred_svm = clf_svm.predict(X)
acc_svm= accuracy_score(Y,pred_svm)*100
print('Accuracy for SImpleVectorMachine:{}'.format(acc_svm))


pred_per= clf_perceptron.predict(X)
acc_per= accuracy_score(Y,pred_per)*100
print('Accuracy for Perceptron:{}'.format(acc_per))


#pred_KNN = clf_KNN.predict(X)
#acc_knn= accuracy_score(Y,pred_per)*100
#print('Accuracy for Knn:{}'.format(acc_knn))

index = np.argmax([acc_svm,acc_per])
classifiers = {0: 'SVM',1: 'Perceptron'}
print('BEST GENDER  CLASSIFIER IS {}'.format(classifiers[index]))