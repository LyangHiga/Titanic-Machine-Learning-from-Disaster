#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC


from sklearn.model_selection import cross_val_score


def plot_hist(titanic):
	titanic.hist(bins=50, figsize=(20,15))
	plt.savefig("attribute_histogram_plots")
	plt.show()

def plot_hist_feat_surv(feature):
	g = sns.FacetGrid(train, col='Survived')
	g.map(plt.hist, feature, bins=20)
	g.add_legend();
	plt.savefig(feature)
	plt.show()


#just import train and test set 
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_test = test.copy()

#convert male to 0 and female to 1
for dataset in train, test:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#for Embarked there are two missing values, we'll put the most common
freq_port = train.Embarked.dropna().mode()[0]	
for dataset in train, test:
	dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
	dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#For age we also have missing values
#for now lets use the median
for dataset in train, test:
	dataset["Age"] = dataset["Age"].fillna(dataset.loc[:,"Age"].median())

#add a new feature Title
for dataset in train, test:
	dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in train, test:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


#separe features from labels
titanic = train.copy()
titanic = titanic.drop("Survived", axis=1)
titanic_labels = train["Survived"].copy()
#print(titanic.head())
#print(titanic_labels.head())
print(titanic.info())
print(titanic.describe())

#plot_hist(titanic)


#for now we'll ignore name, ticket and cabin
titanic = titanic.drop("Name",axis=1)
titanic = titanic.drop("Ticket",axis=1)
titanic = titanic.drop("Cabin",axis=1)
titanic = titanic.drop("PassengerId",axis=1)
print(titanic.info())
#print(titanic.head())

#lets look to correlations
corr_matrix = train.corr()
print(corr_matrix["Survived"].sort_values(ascending=False))
#we can also get feature correlations by pivoting categorical features with our label
print(train[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived", ascending=False))
print(train[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived", ascending=False))
print(train[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived", ascending=False))
print(train[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived", ascending=False))
print(train[["Embarked","Survived"]].groupby(["Embarked"],as_index=False).mean().sort_values(by="Survived", ascending=False))
print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

#plot_hist_feat_surv("Age")
#plot_hist_feat_surv("Fare")
#plot_hist_feat_surv("Pclass")
#plot_hist_feat_surv("Sex")
#plot_hist_feat_surv("Embarked")
#plot_hist_feat_surv("Title")

print(titanic.info())
print(titanic.head())

sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
sgd_clf.fit(titanic, titanic_labels)

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(titanic, titanic_labels)

svc = SVC()
svc.fit(titanic, titanic_labels)

acc_sgd = round(sgd_clf.score(titanic, titanic_labels) * 100, 2)
acc_forest = round(forest_clf.score(titanic, titanic_labels) * 100, 2)
acc_svc = round(svc.score(titanic, titanic_labels) * 100, 2)

print("sgd Accuracy in train_set before scaler: ",acc_sgd)
print("RF Accuracy in train_set before scaler: ",acc_forest)
print("svc Accuracy in train_set before scaler: ",acc_svc)


print('sgd_clf Accuracy using CV with 10 folds: ', cross_val_score(sgd_clf, titanic, titanic_labels, cv=10, scoring="accuracy"))
print('forest Accuracy using CV with 10 folds: ', cross_val_score(forest_clf, titanic, titanic_labels, cv=10, scoring="accuracy"))

#Feature Scaling
#RN just Age ang Fare arent a categorical feature and Fare has a huge scaling 
scaler = StandardScaler()
titanic_scaled = titanic.copy()
titanic_scaled["Age"] = scaler.fit_transform(titanic["Age"].values.reshape(-1,1))
titanic_scaled["Fare"] = scaler.fit_transform(titanic["Fare"].values.reshape(-1,1))
print(titanic_scaled.info())
print(titanic_scaled.head())

sgd_clf.fit(titanic_scaled, titanic_labels)
forest_clf.fit(titanic_scaled, titanic_labels)
svc.fit(titanic_scaled, titanic_labels)

acc_sgd = round(sgd_clf.score(titanic, titanic_labels) * 100, 2)
acc_forest = round(forest_clf.score(titanic, titanic_labels) * 100, 2)
acc_svc = round(svc.score(titanic, titanic_labels) * 100, 2)
print("sgd Accuracy in train_set after scaler: ",acc_sgd)
print("RF Accuracy in train_set after scaler: ",acc_forest)
print("svc Accuracy in train_set after scaler: ",acc_svc)

print('sgd_clf Accuracy using CV with 10 folds: ', cross_val_score(sgd_clf, titanic, titanic_labels, cv=10, scoring="accuracy"))
print('forest Accuracy using CV with 10 folds: ', cross_val_score(forest_clf, titanic, titanic_labels, cv=10, scoring="accuracy"))

#instead of using Stander Scaler we can use Bands to categorize
titanic_bands = titanic.copy()
train['AgeBand'] = pd.cut(train['Age'], 5)
print(train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

titanic_bands.loc[ titanic_bands['Age'] <= 16, 'Age'] = 0
titanic_bands.loc[(titanic_bands['Age'] > 16) & (titanic_bands['Age'] <= 32), 'Age'] = 1
titanic_bands.loc[(titanic_bands['Age'] > 32) & (titanic_bands['Age'] <= 48), 'Age'] = 2
titanic_bands.loc[(titanic_bands['Age'] > 48) & (titanic_bands['Age'] <= 64), 'Age'] = 3
titanic_bands.loc[ titanic_bands['Age'] > 64, 'Age']

train['FareBand'] = pd.qcut(train['Fare'], 4)
print(train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

titanic_bands.loc[ titanic_bands['Fare'] <= 7.91, 'Fare'] = 0
titanic_bands.loc[(titanic_bands['Fare'] > 7.91) & (titanic_bands['Fare'] <= 14.454), 'Fare'] = 1
titanic_bands.loc[(titanic_bands['Fare'] > 14.454) & (titanic_bands['Fare'] <= 31), 'Fare']   = 2
titanic_bands.loc[ titanic_bands['Fare'] > 31, 'Fare'] = 3
titanic_bands['Fare'] = titanic_bands['Fare'].astype(int)

print(titanic_bands.info())
print(titanic_bands.head())

sgd_clf.fit(titanic_bands, titanic_labels)
forest_clf.fit(titanic_bands, titanic_labels)
svc.fit(titanic_bands, titanic_labels)


acc_sgd = round(sgd_clf.score(titanic_bands, titanic_labels) * 100, 2)
acc_forest = round(forest_clf.score(titanic_bands, titanic_labels) * 100, 2)
acc_svc = round(svc.score(titanic_bands, titanic_labels) * 100, 2)
print("sgd Accuracy in train_set bands: ",acc_sgd)
print("RF Accuracy in train_set bands: ",acc_forest)
print("svc Accuracy in train_set bands: ",acc_svc)


test = test.drop("Name",axis=1)
test = test.drop("Ticket",axis=1)
test = test.drop("Cabin",axis=1)
test = test.drop("PassengerId",axis=1)
#test["Age"] = scaler.fit_transform(test["Age"].values.reshape(-1,1))
#test["Fare"] = scaler.fit_transform(test["Fare"].values.reshape(-1,1))

test.loc[ test['Age'] <= 16, 'Age'] = 0
test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1
test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2
test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3
test.loc[ test['Age'] > 64, 'Age']

test.loc[ test['Fare'] <= 7.91, 'Fare'] = 0
test.loc[(test['Fare'] > 7.91) & (test['Fare'] <= 14.454), 'Fare'] = 1
test.loc[(test['Fare'] > 14.454) & (test['Fare'] <= 31), 'Fare']   = 2
test.loc[ test['Fare'] > 31, 'Fare'] = 3

test["Fare"] = test["Fare"].fillna(train.loc[:,"Fare"].median())	

print(test.info())
print(test.head())

Y_pred = forest_clf.predict(test)

submission = pd.DataFrame({
        "PassengerId": X_test["PassengerId"],
        "Survived": Y_pred
    })

submission.to_csv('../Titanic-Machine-Learning-from-Disaster/submission.csv', index=False)



