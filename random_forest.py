import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# from sklearn import linear_model

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
df = pd.read_csv('train.csv')
num = df.median().index
cat = [i for i in df.columns if i not in num]
for i in cat:
    df[i].fillna(df[i].value_counts().index[0], inplace=True)

for i in num:
    df[i].fillna(df[i].median(), inplace=True)
input_age = 45
fare = df['Fare'].astype(int)
age = df['Age'].astype(int)
df.drop(['Fare'], axis=1)
df_new = df.drop(['Age'], axis=1)
df_new['PassengerId'] = df['PassengerId']
df_new['Survived'] = df['Survived']
df_new['Pclass'] = df['Pclass']
df_new['SibSp'] = df['SibSp']
df_new['Parch'] = df['Parch']
df_new['Fare'] = fare
df_new['Age'] = age
columns = df.columns
X = df_new[['PassengerId', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
y = df_new['Survived']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)
X_test['Predection'] = Y_prediction
age_group = X_test.groupby('Age')
listt = age_group.get_group(input_age)['Predection']
max = listt.value_counts().max()

num = listt.value_counts()
if num[0] == max:
    print("person with age {} will not survive".format(input_age))
else:
    print("person with age {} will survive".format(input_age))


