#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn import linear_model 
from sklearn.metrics import confusion_matrix,r2_score,roc_curve,auc,precision_recall_curve,average_precision_score,mean_squared_error


# In[2]:


songs = pd.read_csv('data_updated.csv')


# In[3]:


del songs['Unnamed: 0']
del songs['id']
del songs['release_date']
del songs['artists']
del songs['name']
del songs['emotion']
songs['language'] = np.where(songs['language']=='English',1,0)
songs['popularity'] = np.where(songs['popularity']>=33, 1, 0)
songs['duration_ms'] = songs['duration_ms']/60000
songs.rename(columns={'duration_ms':'duration'})


# ## Neural Network

# In[4]:


import warnings
warnings.filterwarnings('ignore')


# In[5]:


X = songs[songs.columns[0:28]].values
Y = songs[songs.columns[28]].values


# In[6]:


from sklearn.neural_network import MLPClassifier


# In[7]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.20, random_state = 0)


# In[8]:


from sklearn.model_selection import GridSearchCV
parameters = {
    'learning_rate':["constant", "invscaling", "adaptive"],
    'solver': ('sgd','lbfgs','adam'),
    'activation': ('logistic','tanh','relu'),
    'hidden_layer_sizes': ((30,),(60,),(80,)),
    'max_iter': (500, 1000)
}
gs = GridSearchCV(estimator = MLPClassifier(), param_grid=parameters,cv=5)
gs.fit(X, Y)
print(gs.best_score_)
print(gs.best_params_)


# In[9]:


clf = MLPClassifier(solver='adam', hidden_layer_sizes=(60,), max_iter = 1000, 
                    activation='relu',
                    learning_rate='adaptive')
clf.fit(xtrain,ytrain)
predictions = clf.predict(xtest)
actuals = ytest


# In[25]:


p_1 = clf.predict(xtrain)
p_2 = clf.predict(xtest)


# In[13]:


tp=tn=fp=fn=0
for i in range(len(actuals)):
    a_class=p_class=0
    if int(actuals[i]== 0):
        a_class = 1 
    if int(predictions[i]== 0):
        p_class = 1
    if a_class == 1 and p_class == 1:
        tp +=1
    elif a_class == 1 and p_class == 0:
        fn +=1
    elif a_class == 0 and p_class == 0:
        tn +=1
    elif a_class == 0 and p_class == 1:
        fp +=1
print(tp,tn,fp,fn)
nn_accuracy = ((tp+tn)*100/(tp+tn+fp+fn))
print(nn_accuracy)


# In[26]:


from sklearn.metrics import mean_squared_error
mean_squared_error(ytest, p_2,squared=False)


# In[27]:


mean_squared_error(ytrain, p_1,squared=False)


# In[28]:


from sklearn.metrics import r2_score
r2_score(ytest, p_2)


# In[29]:


r2_score(ytrain, p_1)


# In[ ]:




