from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

df = pd.read_csv('/Users/preethamkumarmadupuri/PycharmProjects/PlayTennis/Data/play-tennis/play_tennis.csv')

lb = LabelEncoder()
df['outlook_'] = lb.fit_transform(df['outlook'])
df['temp_'] = lb.fit_transform(df['temp'] )
df['humidity_'] = lb.fit_transform(df['humidity'] )
df['wind_'] = lb.fit_transform(df['wind'] )
df['play_'] = lb.fit_transform(df['play'] )
X = df.iloc[:,5:9]
Y = df.iloc[:,9]

#X_train, X_test , y_train,y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

clf_entropy = DecisionTreeClassifier(criterion='entropy')
clf_entropy.fit(X, Y)
#clf_entropy.fit(X_train.astype(int),y_train.astype(int))
y_pred_en = clf_entropy.predict(X)

print("Accuracy is :{0}".format(accuracy_score(Y.astype(int),y_pred_en) * 100))
