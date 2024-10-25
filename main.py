#machine learning project
import pandas as pd
import numpy as nm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
#dataset manipulation
my_dataset = pd.read_csv('C:\\Users\\hana\\Desktop\\practice programming\\Ml_diabates\\diabetes.csv')
