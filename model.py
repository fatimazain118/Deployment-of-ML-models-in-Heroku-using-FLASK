# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import csv

dataset = pd.read_csv(r'C:\Users\zafatima\OneDrive - Advanced Micro Devices Inc\Documents\Heroku-Demo\Heroku-Demo-master\hiring.csv')
#dataset = csv.reader('hiring.csv')
dataset['experience'].fillna(0, inplace=True) #replacing nan values with 0

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True) #replacing with mean the nan values

X = dataset.iloc[:, :3]  #Creating independent features taking all rows and excluding last col

print(X)
#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))