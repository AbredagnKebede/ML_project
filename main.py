#machine learning project
import pandas as pd
import numpy as nm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
#dataset manipulation
my_dataset = pd.read_csv('C:\\Users\\hana\\Desktop\\practice programming\\Ml_diabates\\diabetes.csv')
x = my_dataset.drop(columns= "Outcome", axis= 1)
y = my_dataset["Outcome"]
scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
#Labling data set
x = standardized_data 
y = my_dataset["Outcome"]
#slpiliting test and Train data
x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
#data training
classifier  = svm.SVC(kernel="linear")
classifier.fit(x_train, y_train)
#evaluation 
x_train_prediction = classifier.predict(x_train)
train_accuracy_prediction = accuracy_score(x_train_prediction, y_train)
x_test_predication = classifier.predict(x_test)
test_accuracy_prediction = accuracy_score(x_test_predication, y_test)
