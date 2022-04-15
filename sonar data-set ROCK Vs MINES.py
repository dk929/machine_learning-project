#!/usr/bin/env python
# coding: utf-8

# ## importing the libraries
# 

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score


# ## data collection and data processing

# In[2]:


#loading the dataset to a pandas dataframe
sonar = pd.read_csv('sonar.all-data-uci.csv')


# In[7]:


sonar.shape


# In[3]:


sonar.head()


# In[4]:


sonar.describe()   #describe --> statistical of the data 


# In[5]:


sonar.info()


# In[8]:


sonar['Label'].value_counts()


# In[9]:


sonar.groupby('Label').mean()


# In[22]:


#separating data and Labels
X = sonar.drop('Label', axis=1)
Y = sonar['Label']


# In[23]:


print(X)
print(Y)


# ## Training and Test data

# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state=1)


# In[27]:


print(X_train, X_test)


# ## model Training --> LogisticRegression

# In[26]:


model = LogisticRegression()


# In[28]:


#training the Logistic Regression model with training data
model.fit(X_train, Y_train)


# ## model Evaluation

# In[29]:


#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[30]:


print('Accuracy on training data:' , training_data_accuracy)


# In[31]:


#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[32]:


print('Accuracy on test data:' , test_data_accuracy)


# ## Making a Predictive System

# In[33]:


# here we will chose random data for prediction , you can chose any random data for predict(Rock or Mine)
input_data = (0.0262,0.0582,0.1099,0.1083,0.0974,0.228,0.2431,0.3771,0.5598,0.6194,0.6333,0.706,0.5544,0.532,0.6479,0.6931,0.6759,0.7551,0.8929,0.8619,0.7974,0.6737,0.4293,0.3648,0.5331,0.2413,0.507,0.8533,0.6036,0.8514,0.8512,0.5045,0.1862,0.2709,0.4232,0.3043,0.6116,0.6756,0.5375,0.4719,0.4647,0.2587,0.2129,0.2222,0.2111,0.0176,0.1348,0.0744,0.013,0.0106,0.0033,0.0232,0.0166,0.0095,0.018,0.0244,0.0316,0.0164,0.0095,0.0078)

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)


if (prediction[0]=='R'):
    print('The object is a Rock')
else:
    print('The object is a Mine')


# In[ ]:




