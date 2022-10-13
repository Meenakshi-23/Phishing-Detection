#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv(r"C:\Users\hello\Downloads\PhishingData.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


#check if there is any null value
data.isnull().sum()


# ## Train-Test Splitting 

# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


y = data['Result']
X= data.drop(['Result'], axis =1)
X.head()


# In[8]:


y.head()


# In[9]:


train_X, test_X, train_y, test_y = train_test_split(X,y, test_size=0.3, random_state=2)


# In[10]:


print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)


# ## Model-1 Logistic Regression

# In[11]:


from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


LR= LogisticRegression()
model_1 = LR.fit(train_X,train_y)


# In[13]:


LR_predict = model_1.predict(test_X)


# In[14]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix


# In[15]:


accuracy_score(LR_predict, test_y)


# In[16]:


print(classification_report(LR_predict, test_y))


# In[17]:


print(confusion_matrix(LR_predict, test_y))


# ## Model_2 Decision Tree

# In[18]:


from sklearn.tree import DecisionTreeClassifier


# In[19]:


dtree= DecisionTreeClassifier()
model_2 = dtree.fit(train_X, train_y)
dtree_predict = model_2.predict(test_X)


# In[20]:


accuracy_score(dtree_predict, test_y)


# In[21]:


print(classification_report(dtree_predict, test_y))


# In[22]:


print(confusion_matrix(dtree_predict, test_y))


# ## Model_3 Random Forest Classifier

# In[23]:


from sklearn.ensemble import RandomForestClassifier


# In[24]:


RF= RandomForestClassifier()
model_3 = RF.fit(train_X, train_y)
RF_predict= model_3.predict(test_X)


# In[25]:


accuracy_score(RF_predict, test_y)


# In[26]:


print(classification_report(RF_predict, test_y))


# In[27]:


print(confusion_matrix(RF_predict, test_y))


# ## Model_4 SVM Classifier

# In[29]:


from sklearn.svm import SVC


# In[31]:


svc= SVC()
model_4 = svc.fit(train_X, train_y)
svc_predict = model_4.predict(test_X)


# In[32]:


accuracy_score(svc_predict, test_y)


# In[33]:


print(classification_report(svc_predict, test_y))


# In[34]:


print(confusion_matrix(svc_predict, test_y))


# ## Model_5 XGBoost Classifier

# In[35]:


from xgboost import XGBClassifier


# In[37]:


xgb= XGBClassifier()
model_5 = xgb.fit(train_X, train_y)
xgb_predict = model_5.predict(test_X)


# In[38]:


accuracy_score(xgb_predict, test_y)


# In[39]:


print(classification_report(xgb_predict, test_y))


# In[40]:


print(confusion_matrix(xgb_predict, test_y))


# ## Comparision of models

# print("LogisticRegression Accuracy:", accuracy_score(LR_predict, test_y), )
# print("DecisionTree Classifier Accuracy:", accuracy_score(dtree_predict, test_y))
# print("RandomForest Classifier Accuracy:", accuracy_score(RF_predict, test_y))
# print("SVM Classifier Accuracy:", accuracy_score(svc_predict, test_y))
# print("XGBoost Classifier Accuracy:", accuracy_score(xgb_predict, test_y))
