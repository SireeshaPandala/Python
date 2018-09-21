from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train_url = 'train.csv'
test_url = 'test.csv'
train = pd.read_csv(train_url)

print(train.columns.values)

print(train.isnull().head())
# Fill missing values with mean column values in the test set
train.fillna(train.mean(), inplace=True)

#Survival based on the sex---> the mean is greater for women
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

print(train.info())

#for simplicity :)
train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
print(train[['PassengerId','Survived','Sex','Age']])

#lets convert sex which is nominal to numerical value :D
labelencoding = LabelEncoder()
labelencoding.fit(train['Sex'])
train['Sex'] = labelencoding.transform(train['Sex'])

#Drop survival column as we do not need the label data for unsupervised learning
X=np.array(train.drop(['Survived'],axis=1))

############## AWESOME after playing with the data lets create the model :)

kmeans = KMeans(n_clusters=2,max_iter=100, random_state=0).fit(X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
import matplotlib.pyplot as plt
