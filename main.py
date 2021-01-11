import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('../iris4feat.dat', header=None, sep=' ', engine='python')

X = df.iloc[:,:4]
print(X.head())

y = df[4]
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
# ~ 96%
print(clf.score(X_test,y_test))

df2 = pd.read_csv('../iris4feat_noclas.dat', header=None, sep=' ', engine='python')
print(clf.predict(df2.iloc[:,:4]))

df3 = pd.read_csv('../irisnewdata.dat', header=None, sep=' ', engine='python')

#Tratamento dos dados
X3 = df3.iloc[:,[3,6,9,12]]
print(X3)

predictions = clf.predict(X3)
func = lambda x : 1 if x==1 else 2
vfunc = np.vectorize(func)

predictions = vfunc(predictions)

occurrences1 = np.count_nonzero(predictions == 1)
print(occurrences1)
print(len(predictions) - occurrences1)