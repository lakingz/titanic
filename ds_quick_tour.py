import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#
# initialization
#
titanic = pd.read_csv('train.csv')
titanic.shape
titanic.describe()
titanic.info()

#
# check correlation
#
titanic_cor = titanic.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis ='columns')
titanic_cor.info()
titanic_cor.corr()

hig_corr = titanic_cor.corr()
hig_corr_features = hig_corr.index[abs(hig_corr["Fare"]) >= 0.25]
hig_corr_features

#
# missing data
#
titanic.isnull().sum()
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean())
titanic['Embarked'] = titanic['Embarked'].fillna(method='bfill')
titanic = titanic.drop(['Cabin'],axis=1)
titanic.isnull().sum()
titanic.head()

titanic = titanic.drop(['Name','Ticket'],axis=1)
titanic.head()
titanic = pd.get_dummies(titanic,columns=['Sex','Embarked'],drop_first=True)
titanic.head()

X = titanic.drop(['Survived'],axis=1)
y = titanic['Survived']

#
# data splitting
#
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=21)

#
# data normalization
#
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
display(X_train.head())
display(X_test.head())


from sklearn.metrics import accuracy_score
#
# Logistic Regression
#
from sklearn.linear_model import LogisticRegression
clf_logreg = LogisticRegression()
clf_logreg.fit(X_train, y_train)
y_pred = clf_logreg.predict(X_test)

log_train = round(clf_logreg.score(X_train, y_train) * 100, 2)
log_accuracy = round(accuracy_score(y_pred, y_test) * 100, 2)
print("Training Accuracy    :",log_train)
print("Model Accuracy Score :",log_accuracy)
cm_lr = confusion_matrix(y_test, y_pred)
cm_lr
#
# svm
#
from sklearn import svm
clf_svm = svm.SVC()
clf_svm.fit(X_train, y_train)
y_pred = clf_svm.predict(X_test)

svm_train = round(clf_svm.score(X_train, y_train) * 100, 2)
svm_accuracy = round(accuracy_score(y_pred, y_test) * 100, 2)
print("Training Accuracy    :",svm_train)
print("Model Accuracy Score :",svm_accuracy)
cm_svm = confusion_matrix(y_test, y_pred)
cm_svm
#addition metrics
clf_svm.support_vectors_
clf_svm.support_
clf_svm.n_support_
#
# naive_bayes
#
from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
clf_nb.fit(X_train, y_train)
y_pred = clf_nb.predict(X_test)
nb_train = round(clf_nb.score(X_train, y_train) * 100, 2)
nb_accuracy = round(accuracy_score(y_pred, y_test) * 100, 2)
print("Training Accuracy    :",nb_train)
print("Model Accuracy Score :",nb_accuracy)
cm_nb = confusion_matrix(y_test, y_pred)
cm_nb
#
# knn
#
from sklearn.neighbors import NearestCentroid
clf_knn = NearestCentroid()
clf_knn.fit(X_train, y_train)
y_pred = clf_knn.predict(X_test)
knn_train = round(clf_knn.score(X_train, y_train) * 100, 2)
knn_accuracy = round(accuracy_score(y_pred, y_test) * 100, 2)
print("Training Accuracy    :",knn_train)
print("Model Accuracy Score :",knn_accuracy)
cm_knn = confusion_matrix(y_test, y_pred)
cm_knn
#
# tree
#
from sklearn import tree
clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(X_train, y_train)
y_pred = clf_tree.predict(X_test)
tree_train = round(clf_tree.score(X_train, y_train) * 100, 2)
tree_accuracy = round(accuracy_score(y_pred, y_test) * 100, 2)
print("Training Accuracy    :",tree_train)
print("Model Accuracy Score :",tree_accuracy)
cm_tree = confusion_matrix(y_test, y_pred)
cm_tree
#addition metrics
tree.plot_tree(clf_tree)
#
# Gradient boosting tree
#
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import make_hastie_10_2
clf_GTB = HistGradientBoostingClassifier(min_samples_leaf=1)
clf_GTB.fit(X_train, y_train)
y_pred = clf_GTB.predict(X_test)
GTB_train = round(clf_GTB.score(X_train, y_train) * 100, 2)
GTB_accuracy = round(accuracy_score(y_pred, y_test) * 100, 2)
print("Training Accuracy    :",GTB_train)
print("Model Accuracy Score :",GTB_accuracy)
cm_gbt = confusion_matrix(y_test, y_pred)
cm_gbt
#
# random forest
#
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=10)
clf_rf.fit(X_train, y_train)
y_pred = clf_rf.predict(X_test)
rf_train = round(clf_rf.score(X_train, y_train) * 100, 2)
rf_accuracy = round(accuracy_score(y_pred, y_test) * 100, 2)
print("Training Accuracy    :",rf_train)
print("Model Accuracy Score :",rf_accuracy)
cm_rf = confusion_matrix(y_test, y_pred)
cm_rf
#
# xgboosting
#
from sklearn.ensemble import GradientBoostingClassifier
clf_xgb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf_xgb.fit(X_train, y_train)
y_pred = clf_xgb.predict(X_test)
xgb_train = round(clf_xgb.score(X_train, y_train) * 100, 2)
xgb_accuracy = round(accuracy_score(y_pred, y_test) * 100, 2)
print("Training Accuracy    :",xgb_train)
print("Model Accuracy Score :",xgb_accuracy)
cm_xgb = confusion_matrix(y_test, y_pred)
cm_xgb

#
#plot confusion matrix if needed
#
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred, labels = clf_logreg.classes_),
                               display_labels=clf_logreg.classes_)
disp.plot()
plt.show()
####################################

