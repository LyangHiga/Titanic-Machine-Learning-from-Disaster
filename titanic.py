#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import cross_val_score

#TODO refactoring
#TODO pipeline
#TODO Regularization


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

def data_transformation(dset):
	#convert male to 0 and female to 1
	for dataset in dset:
	    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

	#for Embarked there are two missing values, we'll put the most common
	for dataset in dset:
		freq_port = dataset.Embarked.dropna().mode()[0]	
	for dataset in dset:
		dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
		dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

	#For age we also have missing values
	#for now lets use the median
	for dataset in dset:
		dataset["Age"] = dataset["Age"].fillna(dataset.loc[:,"Age"].median())
	for dataset in dset:
		dataset["Fare"] = dataset["Fare"].fillna(dataset.loc[:,"Fare"].median())	

	#add a new feature Title
	for dataset in dset:
		dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

	for dataset in dset:
	    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
	 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

	    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
	    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
	    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

	    dataset['Title'] = dataset['Title'].map(title_mapping)
	    dataset['Title'] = dataset['Title'].fillna(0)

def del_feat(dset):
	dset = dset.drop("Name",axis=1)
	dset = dset.drop("Ticket",axis=1)
	dset = dset.drop("Cabin",axis=1)
	dset = dset.drop("PassengerId",axis=1)
	return dset

def correlations():
	corr_matrix = train.corr()
	print(corr_matrix["Survived"].sort_values(ascending=False))
	#we can also get feature correlations by pivoting categorical features with our label
	print(train[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived", ascending=False))
	print(train[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived", ascending=False))
	print(train[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived", ascending=False))
	print(train[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived", ascending=False))
	print(train[["Embarked","Survived"]].groupby(["Embarked"],as_index=False).mean().sort_values(by="Survived", ascending=False))
	print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

	plot_hist_feat_surv("Age")
	plot_hist_feat_surv("Fare")
	plot_hist_feat_surv("Pclass")
	plot_hist_feat_surv("Sex")
	plot_hist_feat_surv("Embarked")
	plot_hist_feat_surv("Title")

def acc_score(clf,name,x,y):
	acc = round(clf.score(x, y) * 100, 2)
	print(name + " Accuracy : " + str(acc))

def feat_scaling(dset,l_feat):
	scaler = StandardScaler()
	for feat in l_feat:
		dset[feat] = scaler.fit_transform(dset[feat].values.reshape(-1,1))
	return dset

def bands(dset):
	train['AgeBand'] = pd.cut(train['Age'], 5)
	#print(train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))
	
	dset.loc[ dset['Age'] <= 16, 'Age'] = 0
	dset.loc[(dset['Age'] > 16) & (dset['Age'] <= 32), 'Age'] = 1
	dset.loc[(dset['Age'] > 32) & (dset['Age'] <= 48), 'Age'] = 2
	dset.loc[(dset['Age'] > 48) & (dset['Age'] <= 64), 'Age'] = 3
	dset.loc[ dset['Age'] > 64, 'Age']

	train['FareBand'] = pd.qcut(train['Fare'], 4)
	#print(train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

	dset.loc[ dset['Fare'] <= 7.91, 'Fare'] = 0
	dset.loc[(dset['Fare'] > 7.91) & (dset['Fare'] <= 14.454), 'Fare'] = 1
	dset.loc[(dset['Fare'] > 14.454) & (dset['Fare'] <= 31), 'Fare']   = 2
	dset.loc[ dset['Fare'] > 31, 'Fare'] = 3
	dset['Fare'] = dset['Fare'].astype(int)

	return dset

def sub(clf):
	Y_pred = clf.predict(test)

	submission = pd.DataFrame({
	        "PassengerId": X_test["PassengerId"],
	        "Survived": Y_pred
	    })

	submission.to_csv('../Titanic-Machine-Learning-from-Disaster/submission.csv', index=False)


#just import train and test set 
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_test = test.copy()

data_transformation([train, test])

#separe features from labels
titanic = train.copy()
titanic = titanic.drop("Survived", axis=1)
titanic_labels = train["Survived"].copy()
#print(titanic.head())
#print(titanic_labels.head())
#print(titanic.info())
#print(titanic.describe())
#plot_hist(titanic)

titanic = del_feat(titanic)
#print(titanic.info())
#print(titanic.describe())

#lets look to correlations
#correlations()


sgd_clf = SGDClassifier(loss = 'modified_huber',max_iter=5, tol=-np.infty, random_state=42)
sgd_clf.fit(titanic, titanic_labels)

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(titanic, titanic_labels)

svc = SVC(probability=True)
svc.fit(titanic, titanic_labels)

log_clf = LogisticRegression()
log_clf.fit(titanic, titanic_labels)

acc_score(sgd_clf, "sgd_clf before scaler",titanic,titanic_labels)
acc_score(forest_clf,"RF before scaler",titanic,titanic_labels )
acc_score(svc,"svc before scaler",titanic,titanic_labels )
acc_score(log_clf,"log_clf before scaler",titanic,titanic_labels )

#print('sgd_clf Accuracy using CV with 10 folds: ', cross_val_score(sgd_clf, titanic, titanic_labels, cv=10, scoring="accuracy"))
#print('forest Accuracy using CV with 10 folds: ', cross_val_score(forest_clf, titanic, titanic_labels, cv=10, scoring="accuracy"))

#Feature Scaling
#RN just Age ang Fare arent a categorical feature and Fare has a huge scaling 
scaler = StandardScaler()
titanic_scaled = titanic.copy()
l_feat = ["Age","Fare"]
titanic_scaled = feat_scaling(titanic_scaled,l_feat)

sgd_clf.fit(titanic_scaled, titanic_labels)
forest_clf.fit(titanic_scaled, titanic_labels)
svc.fit(titanic_scaled, titanic_labels)
log_clf.fit(titanic_scaled, titanic_labels)

acc_score(sgd_clf, "sgd_clf after scaler",titanic_scaled,titanic_labels)
acc_score(forest_clf,"RF after scaler",titanic_scaled,titanic_labels )
acc_score(svc,"svc after scaler",titanic_scaled,titanic_labels )
acc_score(log_clf,"log_clf after scaler",titanic_scaled,titanic_labels )

#print('sgd_clf Accuracy using CV with 10 folds: ', cross_val_score(sgd_clf, titanic, titanic_labels, cv=10, scoring="accuracy"))
#print('forest Accuracy using CV with 10 folds: ', cross_val_score(forest_clf, titanic, titanic_labels, cv=10, scoring="accuracy"))

#instead of using Stander Scaler we can use Bands to categorize
titanic_bands = titanic.copy()
train['AgeBand'] = pd.cut(train['Age'], 5)
titanic_bands = bands(titanic_bands)

sgd_clf.fit(titanic_bands, titanic_labels)
forest_clf.fit(titanic_bands, titanic_labels)
svc.fit(titanic_bands, titanic_labels)
log_clf.fit(titanic_bands, titanic_labels)

acc_score(sgd_clf, "sgd_clf bands",titanic_bands,titanic_labels)
acc_score(forest_clf,"RF bands",titanic_bands,titanic_labels )
acc_score(svc,"svc bands",titanic_bands,titanic_labels )
acc_score(log_clf,"log_clf bands",titanic_bands,titanic_labels )
'''

bag_clf = BaggingClassifier(
	RandomForestClassifier(n_estimators=100, random_state=42), n_estimators=500,
	max_samples=100, bootstrap=True, n_jobs=-1
)
bag_clf.fit(titanic_bands, titanic_labels)
acc_score(bag_clf,"bag_clf bands",titanic_bands,titanic_labels )
'''

voting_clf_hard = VotingClassifier(
	estimators = [('svm',svc), ('forest',forest_clf)],
	voting = 'hard'
	)
voting_clf_hard.fit(titanic, titanic_labels)
acc_score(voting_clf_hard,"voting_clf_hard before scaler",titanic,titanic_labels )

voting_clf_soft = VotingClassifier(
	estimators = [('svm',svc), ('forest',forest_clf)],
	voting = 'soft'
	)
voting_clf_soft.fit(titanic, titanic_labels)
acc_score(voting_clf_soft,"voting_clf_soft before scaler",titanic,titanic_labels )

test = del_feat(test)
#test = feat_scaling(test,l_feat)
test = bands(test)

print(test.info())
print(test.head())

sub(voting_clf_soft)
