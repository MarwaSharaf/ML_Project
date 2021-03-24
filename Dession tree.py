# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:26:36 2021

Desion tree

@author: Nosila
"""
#7.0,3.2,4.7,1.4 virccolor =1
#5.1 3.5 1.4 0.2 setosa =0
#6.3,3.3,6.0,2.5 virginica =2

# first import the library
import pandas as pd
from numpy import nan
# import joblib
import joblib as joblib

################################################################################
# datasert load
# dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']
dataset = pd.read_csv(url, names=names)

print(dataset.head())  # it prints rows of data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['species']= le.fit_transform(dataset['species'])

##############################################################################################


# slicing
# x contaion first 4colume and y contain class lable (setosa or vercicolor or virginica) 
X_features_input = dataset.iloc[:, :-1].values  # features[rows, columms]
print(X_features_input)
y_label_output = dataset.iloc[:, 4].values  # labels
# label encoding the data


#################################################################

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_features_input, y_label_output, test_size=0.20, random_state=5)
# x_train = 80% of our features data(input)
# x_test = 20% of our features data(input)
# y_train = 80% of our lable data(output)
# y_test = 20 % of pur lable data(output)

###########################################################################################

# imported the algorithms from library
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
# to train the model you have to use the function of "fit()"
# while traininf we only pass the 80 percent of our data
classifier.fit(X_train, y_train) # X_train = features  #y_train= lable
# now we have to take prediction on testing data
y_pred = classifier.predict(X_test) #here we only pass the features


#save model 
filename = 'Decisiontree_model.sav'
joblib.dump(classifier,'Decisiontree_model.sav')
 
# some time later...
 
# load the model from disk
loaded_model = joblib.load('Decisiontree_model.sav')
print('Accuracy of loaded model')
result = loaded_model.score(X_test, y_test)
print(result)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(y_pred)
from sklearn.metrics import accuracy_score

print('Accuracy Score: ', accuracy_score(y_pred, y_test))  # y_pred is the output

from sklearn.metrics import f1_score

f1_metric = f1_score(y_test, y_pred, average='macro')
print("f1 score macro:", f1_metric)

from sklearn.metrics import f1_score

f1_metric_micro = f1_score(y_test, y_pred, average='micro')
print("f1 score micro:", f1_metric_micro)

#4.6  4.6          3.1           1.5          0.2 

# print(tree.plot_tree(classifier))
from mlxtend.evaluate import bias_variance_decomp
mse, bias, var = bias_variance_decomp(classifier, X_train, y_train, X_test, y_test, num_rounds=200, random_seed=1)
# summarize results
print('Bias: %.3f' % bias)
print('Variance: %.3f' % var)
from sklearn.model_selection import cross_val_score
# clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(classifier.fit(X_train, y_train), X_features_input,y_label_output, cv=5)
print('Cross Validation')
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
   
###############################################################



#take input from the loaded model

input_sepal_length = float(input("Enter sepal length: "))
input_sepal_width = float(input("Enter sepal width:"))
input_petal_length = float(input("Enter petal Length: "))
input_petal_width = float(input("Enter petal width: "))
output = loaded_model.predict([[input_sepal_length, input_sepal_width,input_petal_length,input_petal_width ]])
print(output)