from sklearn import datasets, metrics#The import dimension (or metric) receives this additional data you are uploading
from sklearn import cross_validation as cv# The scoring parameter: defining model evaluation rules for details. .... from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB#we can use naives by using this

# Loading the dataset
irisdataset = datasets.load_iris()

#
model = GaussianNB()
# # getting the data and response of the dataset
x = irisdataset.data
y = irisdataset.target
# model.fit(x,y)
# print(model)
# model.predict(x,y)

splits = cv.train_test_split(x,y,test_size=0.4)#splits the data into test and train for x, and y
X_train, X_test, y_train, y_test =splits
#print(splits)
model.fit(X_train,y_train)#fit the training values of x & y to  the model
expected = y_test
predicted = model.predict(X_test)#stores the predicted x value into predicted
print("----")
print(metrics.classification_report(expected,predicted))#gives the expected and predicted value
print("----")
print(metrics.confusion_matrix(expected,predicted))#gets the predicted and expected values into confusion matrix
print("----")

print(metrics.f1_score(expected,predicted, average='micro'))#gives us the accurracy