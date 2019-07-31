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
def train_and_eval(ks, layers, size_of_layers, epochs):


    for k in ks:
        k= int(k)

        census_data = pd.read_csv('ano-data/titanic/'+str(k)+'_titanic.csv',header = 0,sep=',',engine='python',na_values='?')
        census_test = pd.read_csv('ano-data/titanic/'+str(1)+'_titanic_test.csv',header = 0,sep=',',engine='python',na_values='?')

        census_data.drop(census_data.columns[0], axis=1,inplace = True)
        census_test.drop(census_test.columns[0], axis=1,inplace = True)

        data_feature = census_data.drop(columns=['Survived']).copy()
        data_label = census_data['Survived'].copy()

        test_feature = census_test.drop(columns=['Survived']).copy()
        test_label = census_test['Survived'].copy()
        
        for i in range(data_label.shape[0]):
            data_label.iloc[i]=int("1" in str(data_label.iloc[i]))
        for i in range(test_label.shape[0]):
            test_label.iloc[i]=int("1" in str(test_label.iloc[i]))
        
        import keras
        from keras.models import Sequential
        from keras.layers import Dense
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        import time

        data_feature, val_feature, data_label, val_label = train_test_split(data_feature, data_label, test_size=0.15, random_state=42,stratify = data_label)

        #SIZES = [4,8,16]
        SIZES = [int(size_of_layers)]
        times = np.zeros(len(SIZES))
        accs = np.zeros(len(SIZES))
        NAVG = 3
        for i in range(len(SIZES)):
            for counter in range(NAVG):
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
                times[i]+=(time.time()-start)/NAVG
                y_pred = classifier.predict(test_feature)
                y_pred = np.array([int(y>0.5) for y in y_pred])
                accs[i]+=Accuracy(y_pred,test_label)/NAVG
        
        plt.figure()
        plt.plot(SIZES,times)
        plt.xlabel("Size of one layer")
        plt.title("Training time (s)")
        plt.savefig('Titanic_traintimes_'+str(k)+".png")
        
        plt.figure()
        plt.plot(SIZES,accs*100)
        plt.xlabel("Size of one layer")
        plt.title("Accuracy (%)")
        plt.savefig('Titanic_accuracy_'+str(k)+".png")

