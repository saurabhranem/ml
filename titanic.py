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
from sklearn.ensemble import RandomForestClassifierim
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

fare = df['Fare'].astype(int)
age = df['Age'].astype(int)
df.drop(['Fare'], axis=1)
# df.drop(['Age'], axis=1)
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

# Linear regression
lm = LinearRegression()
lin_reg_model = lm.fit(X_train, Y_train)
predictions = lm.predict(X_test)
# plt.scatter(y_test, predictions)
# print(lin_reg_model.score(X_test, y_test))
# print(round(lin_reg_model.score(X_test, y_test) * 100, 2))
acc_linear = round(lin_reg_model.score(X_train, Y_train) * 100, 2)

# Stochastic Gradient Descent (SGD)
sgd = linear_model.SGDClassifier(max_iter=5000, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
sgd.score(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

# #Logestic Regression
# logreg = LogisticRegression(max_iter=50000)
# logreg.fit(X_train, Y_train)
# Y_predl = logreg.predict(X_test)
# acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

#KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_predk = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

#Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_predg = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

#Prcwptron
perceptron = Perceptron(max_iter=5000)
perceptron.fit(X_train, Y_train)
Y_predp = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

#linear support vendor machine
# linear_svc = LinearSVC(max_iter=500000)
# linear_svc.fit(X_train, Y_train)
# Y_predli = linear_svc.predict(X_test)
# acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

#decision tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_predd = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
results = pd.DataFrame({
    'Model': ['KNN',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent',
              'Decision Tree'],
    'Score': [acc_knn,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
print(result_df.head(9))