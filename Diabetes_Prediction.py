# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 16:48:16 2021

@author: SOUMICK KARMAKAR
"""

import pandas as pd
import seaborn as sc

a=pd.read_csv("train.csv")
b=pd.read_csv("test1.csv")
a=a.drop(["Id"],axis=1)
a=a.drop(["SkinThickness"],axis=1)
a=a.drop(["Insulin"],axis=1)
a=a.drop(["Pregnancies"],axis=1)
b=b.drop(["Pregnancies"],axis=1)
b=b.drop(["SkinThickness"],axis=1)
b=b.drop(["Insulin"],axis=1)
b=b.drop(["Id"],axis=1)
b=b.drop(["split"],axis=1)
print(a.head())
print(b.head())

print(a.isna().sum())

sc.pairplot(a, hue ='Outcome')

x_train=a.iloc[:,:-1].values
x_test=b.iloc[:,:-1].values
y_train=a.iloc[:,-1].values
y_test=b.iloc[:,-1].values

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 1) 
knn.fit(x_train, y_train) 
y_pred=knn.predict(x_test)
print(knn.score(x_test,y_test))

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators =150)
clf.fit(x_train, y_train)
clf.score(x_test,y_test)
pred=clf.predict(x_test)
print(clf.score(x_test,y_test))
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,y_pred))
print(pred)

