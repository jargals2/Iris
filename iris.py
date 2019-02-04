import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pickle

#assign column names to our dataset
iris_data = pd.read_csv('iris_data.txt', sep = ',', header = None)
iris_data.columns = ['sep_len', 'sep_wid', 'pet_len', 'pet_wid', 'class']

#split the dataset into training and testing sets 0.75:0.25
train_data, test_data = train_test_split(iris_data, test_size=0.25, random_state = 100)

train_data_X = train_data.drop(['class'],axis=1)
train_data_y = train_data['class']

test_data_X = test_data.drop(['class'],axis=1)
test_data_y = test_data['class']


#choose 10 as num of neighbers    
knn_model = KNeighborsClassifier(10)
knn_model.fit(train_data_X, train_data_y)


#save the model in a file
pickle.dump(knn_model, open('knn_model.pkl','wb'))



