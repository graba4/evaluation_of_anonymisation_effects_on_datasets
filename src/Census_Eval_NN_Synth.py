#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import time
import graphviz
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time


# In[2]:


def Accuracy(predicted, truth):
    correct = 0
    for i in range(len(predicted)):
        if predicted[i]==truth.iloc[i]:
            correct +=1
    return correct/len(predicted)


# In[3]:


#ks = [1,2,4,8,16,32,64]


# In[4]:
def train_and_eval(layers, size_of_layers, epochs):


    census_data = pd.read_csv('census/out/correlated_attribute_mode/sythetic_data.csv',header = 0,sep=',',engine='python',na_values='?')
    census_test = pd.read_csv('census/adult_test.csv',header = 0,sep=', ',engine='python',na_values='?')
    
    data_feature = census_data.drop(columns=['income']).copy()
    data_label = census_data['income'].copy()

    test_feature = census_test.drop(columns=['income']).copy()
    test_label = census_test['income'].copy()

    for i in range(len(test_label)):
        test_label.loc[i]=str(test_label.loc[i]).strip('.')
        
    #Converting labels to 0-1
    for i in range(data_label.shape[0]):
        data_label.iloc[i]=int("<" in data_label.iloc[i])
    for i in range(test_label.shape[0]):
        test_label.iloc[i]=int("<" in test_label.iloc[i])
    
    anonimizeds = ['age', 'education-num','race','native-country','workclass','hours-per-week','sex']
    cols = data_feature.columns.values

    for col in cols:
        start = col.split('_')[0]
        if start not in anonimizeds:
            data_feature.drop(columns=[col],inplace=True)
            test_feature.drop(columns=[col],inplace=True)
            
###############################################################################################################
    sc = StandardScaler()
    data_feature = sc.fit_transform(data_feature)
    test_feature = sc.transform(test_feature)

    data_feature, val_feature, data_label, val_label = train_test_split(data_feature, data_label, test_size=0.15, random_state=42,stratify = data_label)

    #SIZES = [32,64]
    SIZES = [int(size_of_layers)]
    times_scal = np.zeros(len(SIZES))
    accs_scal = np.zeros(len(SIZES))

    for i in range(len(SIZES)):
        classifier = Sequential()
        # Adding the input layer and the first hidden layer
        classifier.add(Dense(output_dim = SIZES[i], init = 'uniform', activation = 'relu', input_dim = data_feature.shape[1]))
        # Adding the second hidden layer
        classifier.add(Dense(output_dim = SIZES[i], init = 'uniform', activation = 'relu'))
        # Adding the output layer
        classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        clb = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=True)]
        start = time.time()
        classifier.fit(data_feature, data_label, batch_size = 8, nb_epoch = int(epochs),validation_data=(val_feature,val_label),callbacks=clb)
        times_scal[i]=time.time()-start
        y_pred = classifier.predict(test_feature)
        y_pred = np.array([int(y>0.5) for y in y_pred])
        accs_scal[i]=Accuracy(y_pred,test_label)
    
    plt.figure()
    plt.plot(SIZES,times_scal,label = "Scaled")
    plt.xlabel("Size of one layer")
    plt.title("Training time (s)")
    plt.legend()
    plt.savefig("CensusIncome_NN_traintime_hpwsex"+"synthetic_data"+".png")
    
    plt.figure()
    plt.plot(SIZES,accs_scal*100,label = "Scaled")
    plt.xlabel("Size of one layer")
    plt.title("Accuracy (%)")
    plt.legend()
    plt.savefig("CensusIncome_NN_Accuracy_hpwsex"+"synthetic_data"+".png")

