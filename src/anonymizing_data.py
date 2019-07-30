#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


names = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","label"]


# In[3]:


census_data = pd.read_csv('./adult.data',header = None,sep=', ',engine='python',na_values='?',names = names)
census_test = pd.read_csv('./adult.test',header = None,sep=', ',engine='python',na_values='?',skiprows=[0],names=names)


# In[4]:


for i in range(census_test.shape[0]):
    census_test.iloc[i,-1]=census_test.iloc[i,-1].strip('.')


# In[5]:


# generalizing ages into bigger groups
for i in range(census_data.shape[0]):
    #print(census_data.iloc[i,0])
    if(census_data.iloc[i,0]<=25):
        census_data.iloc[i,0] = '18-25'
    elif(census_data.iloc[i,0]>25 and census_data.iloc[i,0]<=35):
        census_data.iloc[i,0] = '25-35'
    elif(census_data.iloc[i,0]>35 and census_data.iloc[i,0]<=45):
        census_data.iloc[i,0] = '35-45'
    elif(census_data.iloc[i,0]>45 and census_data.iloc[i,0]<=55):
        census_data.iloc[i,0] = '45-55'
    elif(census_data.iloc[i,0]>55 and census_data.iloc[i,0]<=65):
        census_data.iloc[i,0] = '55-65'
    elif(census_data.iloc[i,0]>65 and census_data.iloc[i,0]<=75):
        census_data.iloc[i,0] = '65-75'
    else:
        census_data.iloc[i,0] = '75+'
    #print(census_data.iloc[i,0])


# In[6]:


# generalizing education_num to bigger groups, we should drop the column 3 which is education only because it 
# is already presented by this number
for i in range(census_data.shape[0]):
    #print(census_data.iloc[i,4])
    if(census_data.iloc[i,4]<=4):
        census_data.iloc[i,4] = '1-4'
    elif(census_data.iloc[i,4]>4 and census_data.iloc[i,4]<=8):
        census_data.iloc[i,4] = '5-8'
    elif(census_data.iloc[i,4]>8 and census_data.iloc[i,4]<=12):
        census_data.iloc[i,4] = '9-12'
    else:
        census_data.iloc[i,4] = '13-16'
    #print(census_data.iloc[i,4])


# In[7]:


# generalizing countries into continents
for i in range(census_data.shape[0]):
    #print(census_data.iloc[i,-2])
    if(census_data.iloc[i,-2] == 'United-States' or
       census_data.iloc[i,-2] == 'Puerto-Rico' or
       census_data.iloc[i,-2] == 'Canada' or 
       census_data.iloc[i,-2] == 'Outlying-US(Guam-USVI-etc)' or
       census_data.iloc[i,-2] == 'Cuba' or
       census_data.iloc[i,-2] == 'Honduras' or 
       census_data.iloc[i,-2] == 'Jamaica' or
       census_data.iloc[i,-2] == 'Mexico' or 
       census_data.iloc[i,-2] == 'Dominican-Republic' or 
       census_data.iloc[i,-2] == 'Haiti' or
       census_data.iloc[i,-2] == 'Guatemala' or
       census_data.iloc[i,-2] == 'Nicaragua' or 
       census_data.iloc[i,-2] == 'El-Salvador' or
       census_data.iloc[i,-2] == 'Trinadad&Tobago'):
        census_data.iloc[i,-2] = 'North-and-Central-America'
    elif(census_data.iloc[i,-2] == 'Cambodia' or
        census_data.iloc[i,-2] == 'India' or
        census_data.iloc[i,-2] == 'Japan' or 
        census_data.iloc[i,-2] == 'China' or
        census_data.iloc[i,-2] == 'Iran' or
        census_data.iloc[i,-2] == 'Philippines' or 
        census_data.iloc[i,-2] == 'Vietnam' or
        census_data.iloc[i,-2] == 'Laos' or
        census_data.iloc[i,-2] == 'Taiwan' or
        census_data.iloc[i,-2] == 'Thailand' or 
        census_data.iloc[i,-2] == 'Hong'):
        census_data.iloc[i,-2] = 'Asia'
    elif(census_data.iloc[i,-2] == 'Columbia' or 
        census_data.iloc[i,-2] == 'Peru' or 
        census_data.iloc[i,-2] == 'Ecuador'):
        census_data.iloc[i,-2] = 'South-America'
    elif(census_data.iloc[i,-2] == 'South'):
        census_data.iloc[i,-2] = 'Africa'
    else:
        census_data.iloc[i,-2] = 'Europe'
    #print(census_data.iloc[i,-2])


# In[8]:


#generalizing race, non-blacks and non-whites to others, pretty sensitive
for i in range(census_data.shape[0]):
    if(census_data.iloc[i,8] == 'Asian-Pac-Islander' or 
       census_data.iloc[i,8] == 'Amer-Indian-Eskimo'):
        census_data.iloc[i,8] = 'Other'


# In[9]:


#generalizing workclass to government, not government and others
for i in range(census_data.shape[0]):
    if(census_data.iloc[i,1] == 'Private' or 
       census_data.iloc[i,1] == 'Self-emp-not-inc' or 
       census_data.iloc[i,1] == 'Self-emp-inc'):
        census_data.iloc[i,1] = 'Not-gov'
    elif(census_data.iloc[i,1] == 'Federal-gov' or 
         census_data.iloc[i,1] == 'Local-gov' or 
         census_data.iloc[i,1] == 'State-gov'):
        census_data.iloc[i,1] = 'Gov'
    else:
        census_data.iloc[i,1] = 'Other'


# In[ ]:




