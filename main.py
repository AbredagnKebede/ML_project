#machine learning project important libraries
import pandas as pd
import numpy as np
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

#Labling dataset
x = standardized_data 
y = my_dataset["Outcome"]

#slpiliting test and Train data
x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

#data training
classifier  = svm.SVC(kernel="linear")
classifier.fit(x_train, y_train)

#evaluation, test 
x_train_prediction = classifier.predict(x_train)
train_accuracy_prediction = accuracy_score(x_train_prediction, y_train)

#train
x_test_predication = classifier.predict(x_test)
test_accuracy_prediction = accuracy_score(x_test_predication, y_test)
print("**********************************************")
print(f"Training accuracy: {train_accuracy_prediction*100}%")
print(f"Testing accuracy: {test_accuracy_prediction*100}%")
print("**********************************************")

#predicating an input
print("Enter patient info in proper order as it was given in training: ",)
input_data = tuple(map(float, input().split(",")))
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardizing the input
std_data = scaler.transform(input_data_reshaped)

# Making a prediction
prediction = classifier.predict(std_data)
print("____________________________________________________")
if prediction[0] == 1:
    print(f' The patient probabilty has diabetes', f"System accuracy is: {test_accuracy_prediction*100}%")
else:
    print(f' The patient probabilty is diabetes free\n', f"System accuracy is: {test_accuracy_prediction*100}%")
print("____________________________________________________")
    