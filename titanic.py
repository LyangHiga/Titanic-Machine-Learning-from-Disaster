#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder


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

#convert male to 0 and female to 1
for dataset in train, test:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#for Embarked there are two missing values, we'll put the most common
freq_port = train.Embarked.dropna().mode()[0]	
for dataset in train, test:
	dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
	dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#separe features from labels
titanic = train.copy()
titanic.drop("Survived", axis=1)
titanic_labels = train["Survived"].copy()
print(titanic.head())
print(titanic_labels.head())
print(titanic.info())
print(titanic.describe())

#plot_hist(titanic)


#for now we'll ignore name, ticket and cabin
titanic = titanic.drop("Name",axis=1)
titanic = titanic.drop("Ticket",axis=1)
titanic = titanic.drop("Cabin",axis=1)
print(titanic.info())
print(titanic.head())

#lets look to correlations
corr_matrix = train.corr()
print(corr_matrix["Survived"].sort_values(ascending=False))
#we can also get feature correlations by pivoting categorical features with our label
print(train[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived", ascending=False))
print(train[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived", ascending=False))
print(train[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived", ascending=False))
print(train[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived", ascending=False))
print(train[["Embarked","Survived"]].groupby(["Embarked"],as_index=False).mean().sort_values(by="Survived", ascending=False))

#plot_hist_feat_surv("Age")
#plot_hist_feat_surv("Fare")
#plot_hist_feat_surv("Pclass")
plot_hist_feat_surv("Sex")
plot_hist_feat_surv("Embarked")






