#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[2]:


credit_card_data = pd.read_csv('creditcard.csv')
credit_card_data.head()


# In[3]:


credit_card_data.sample()


# In[4]:


# dataset informations
credit_card_data.info()


# In[5]:


# checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[6]:


# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()


# In[7]:


legit = credit_card_data[credit_card_data.Class==0]
fraud = credit_card_data[credit_card_data['Class']==1]
fraud['Class']


# In[8]:


legit.Amount.describe()


# In[9]:


fraud.Amount.describe()


# In[10]:


credit_card_data.groupby('Class').mean()


# In[11]:


legit_sample = legit.sample(n=492)
new_df = pd.concat([legit_sample,fraud],axis=0)
new_df


# In[12]:


new_df['Class'].value_counts()


# In[13]:


new_df.groupby('Class').mean()


# In[14]:


X = new_df.drop(columns='Class', axis=1)
Y = new_df['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[15]:


model=LogisticRegression()
# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)


# In[16]:


random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, Y_train)
random_forest_train_accuracy = accuracy_score(random_forest_model.predict(X_train), Y_train)
random_forest_test_accuracy = accuracy_score(random_forest_model.predict(X_test), Y_test)
print('Random Forest - Training Accuracy:', random_forest_train_accuracy)
print('Random Forest - Test Accuracy:', random_forest_test_accuracy)


# In[17]:


svm_model = SVC()
svm_model.fit(X_train, Y_train)
svm_train_accuracy = accuracy_score(svm_model.predict(X_train), Y_train)
svm_test_accuracy = accuracy_score(svm_model.predict(X_test), Y_test)
print('SVM - Training Accuracy:', svm_train_accuracy)
print('SVM - Test Accuracy:', svm_test_accuracy)


# In[19]:


# Train and evaluate KNN model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, Y_train)
knn_train_accuracy = accuracy_score(knn_model.predict(X_train), Y_train)
knn_test_accuracy = accuracy_score(knn_model.predict(X_test), Y_test)
print('KNN - Training Accuracy:', knn_train_accuracy)
print('KNN - Test Accuracy:', knn_test_accuracy)






# In[ ]:




