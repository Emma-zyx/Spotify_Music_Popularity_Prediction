#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import pylab 
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn
import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn import tree, preprocessing
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, make_scorer,recall_score,precision_score,roc_auc_score,roc_curve, classification_report, confusion_matrix


# In[2]:


df_new = pd.read_csv('data_updated.csv')
#from sklearn.preprocessing import LabelEncoder
df_new.loc[(df_new.language == 'English'),'language']= 1
df_new.loc[(df_new.language == 'Other Languages'),'language']= 0
df_new['duration'] = df_new['duration_ms']/60000


# In[3]:


df_new = df_new.drop(['Unnamed: 0','duration_ms','id','artists','name','release_date','emotion'], axis = 1)


# In[4]:


df_new.head()


# In[5]:


df_new['pop'] = np.where(df_new['popularity'] >= 33, 1, 0)


# # First set of features:
# 
# valence, year, acousticness, danceability, energy, explicit, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, vocab, avg_word_len, uniqueness, language, duration

# In[6]:


from sklearn.model_selection import train_test_split
train_1, test_1 = train_test_split(df_new, test_size = 0.3, random_state=0)
x_train_1 = train_1.drop(['popularity','pop'], axis=1)
y_train_1 = train_1['pop']
x_test_1 = test_1.drop(['popularity','pop'], axis=1)
y_test_1 = test_1['pop']
scaler=StandardScaler()
scaler.fit(x_train_1)
x_train_1 = scaler.transform(x_train_1)
x_test_1 = scaler.transform(x_test_1)


# In[7]:


tree_mod= tree.DecisionTreeClassifier(max_depth=3)
treemod1 = tree_mod.fit(x_train_1, y_train_1)

xgb_rb = xgb.XGBClassifier()
xgb_mod1 = xgb_rb.fit(x_train_1, y_train_1)

rf = RandomForestClassifier(n_jobs=8)
rf_mod1 = rf.fit(x_train_1, y_train_1)


# In[8]:


def train_model(model,x_train,y_train,x_test,y_test):
    pred = model.predict(x_test)
    model = model.fit(x_train, y_train)
    result = dict()
    result['accuracy_score']=accuracy_score(y_test, pred)
    result['recall_score']=recall_score(y_test, pred)
    result['f1_score']=f1_score(y_test, pred)
    result['roc_auc_score']=roc_auc_score(y_test, pred)
    # confusion matrix
    print('confusion_matrix')
    print(pd.DataFrame(confusion_matrix(y_test, pred)))
    return result


# In[9]:


train_model(treemod1,x_train_1,y_train_1,x_test_1,y_test_1)


# In[10]:


train_model(xgb_mod1,x_train_1,y_train_1,x_test_1,y_test_1)


# In[11]:


train_model(rf_mod1,x_train_1,y_train_1,x_test_1,y_test_1)


# # Second set of features:
# 
# year, key, mode, tempo, vocab, avg_word_len, uniqueness, language, duration

# In[12]:


df_2 = df_new[['year','key','mode','tempo','#vocab','avg_word_len','uniqueness','language','comm','negative','anticipation',
               'anger','disgust','fear','joy','positive','sadness','surprise','trust','duration','pop']]
train_2, test_2 = train_test_split(df_2, test_size = 0.3, random_state=0)
x_train_2 = train_2.drop(['pop'], axis=1)
y_train_2 = train_2['pop']
x_test_2 = test_2.drop(['pop'], axis=1)
y_test_2 = test_2['pop']
scaler=StandardScaler()
scaler.fit(x_train_2)
x_train_2 = scaler.transform(x_train_2)
x_test_2 = scaler.transform(x_test_2)


# In[13]:


treemod2 = tree_mod.fit(x_train_2, y_train_2)

xgb_mod2 = xgb_rb.fit(x_train_2, y_train_2)

rf_mod2 = rf.fit(x_train_2, y_train_2)


# In[14]:


train_model(treemod2,x_train_2,y_train_2,x_test_2,y_test_2)


# In[15]:


train_model(xgb_mod2,x_train_2,y_train_2,x_test_2,y_test_2)


# In[16]:


train_model(rf_mod2,x_train_2,y_train_2,x_test_2,y_test_2)


# # Third set of features
# valence, year, acousticness, danceability, energy, explicit, instrumentalness, key, liveness, loudness, mode, speechiness, tempo

# In[17]:


df_3 = df_new[['year','key','mode','tempo','duration', 'pop']]
train_3, test_3 = train_test_split(df_3, test_size = 0.3, random_state=0)
x_train_3 = train_3.drop(['pop'], axis=1)
y_train_3 = train_3['pop']
x_test_3 = test_3.drop(['pop'], axis=1)
y_test_3 = test_3['pop']
scaler=StandardScaler()
scaler.fit(x_train_3)
x_train_3 = scaler.transform(x_train_3)
x_test_3 = scaler.transform(x_test_3)


# In[18]:


treemod3 = tree_mod.fit(x_train_3, y_train_3)

xgb_mod3 = xgb_rb.fit(x_train_3, y_train_3)

rf_mod3 = rf.fit(x_train_3, y_train_3)


# In[19]:


train_model(treemod3,x_train_3,y_train_3,x_test_3,y_test_3)


# In[20]:


train_model(xgb_mod3,x_train_3,y_train_3,x_test_3,y_test_3)


# In[21]:


train_model(rf_mod3,x_train_3,y_train_3,x_test_3,y_test_3)


# In[22]:


# parameters = {
#     'max_depth': [10, 15, 20],
#     'min_child_weight': [2, 5, 10],
#     'subsample': [0.7, 0.8, 0.85],
#     'colsample_bytree': [0.6, 0.7, 0.8]
# }


# In[23]:


param_grid = dict(
    max_depth = [4, 5, 6, 7],
    learning_rate = np.linspace(0.03, 0.3, 5),
    n_estimators = [100, 200, 300]
)


# # Grid Search for parameter tuning

# In[24]:


#sklearn.metrics.SCORERS.keys()


# In[25]:


grid = GridSearchCV(xgb_mod1,param_grid,cv=5, n_jobs=8,scoring = 'accuracy')
grid.fit(x_train_1, y_train_1) 


# In[26]:


xgb_1_new = grid.best_estimator_
xgb_1_new.fit(x_train_1, y_train_1)

print("Boosted Tree Training Acc",xgb_1_new.score(x_train_1,y_train_1))
print("Boosted Tree Testing Acc",xgb_1_new.score(x_test_1,y_test_1))


# In[27]:


#params_rf = {"n_estimators": [50, 200, 300, 350, 400, 600],
#              'criterion': ['gini', 'entropy'],
#              'max_depth': list(range(1, 9, 2)),
#              'bootstrap': [True, False]}


# In[28]:


#grid_f = GridSearchCV(rf_1, params_space)
#grid.fit(X_train, y_train)


# In[29]:


x_train_1_df = pd.DataFrame(columns=train_1.drop(['popularity','pop'], axis=1).columns, data=x_train_1)


# In[30]:


import shap
shap.initjs() 
explainer = shap.TreeExplainer(xgb_1_new)
shap_values = explainer.shap_values(x_train_1)
for i in range(10):
    shap.force_plot(explainer.expected_value, shap_values[i,:], x_train_1_df.iloc[i,:])


# In[31]:


shap.summary_plot(shap_values, x_train_1_df)


# In[32]:


def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
    return df

# plotting a feature importance dataframe (horizontal barchart)
def var_imp_plot(imp_df, title):
    imp_df.columns = ['feature', 'feature_importance']
    sns.barplot(x = 'feature_importance', y = 'feature', data = imp_df, orient = 'h', color = 'royalblue')        .set_title(title, fontsize = 20)


# # KNN

# In[33]:


k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5,n_jobs = 8, scoring='accuracy')
grid.fit(x_train_1,y_train_1)


# In[34]:


grid.best_score_


# In[35]:


score1 = grid.cv_results_['mean_test_score']
neighbors = list(grid.cv_results_['param_n_neighbors'])


# In[36]:


plt.plot(neighbors, score1)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# In[37]:


k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
grid2 = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5,n_jobs = 8, scoring='accuracy')
grid2.fit(x_train_2,y_train_2)


# In[38]:


grid2.best_score_


# In[39]:


score2 = grid2.cv_results_['mean_test_score']
neighbors = list(grid2.cv_results_['param_n_neighbors'])


# In[40]:


plt.plot(neighbors, score2)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# In[41]:


k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
grid3 = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5,n_jobs = 8, scoring='accuracy')
grid3.fit(x_train_3,y_train_3)


# In[42]:


grid3.best_score_


# In[43]:


score3 = grid3.cv_results_['mean_test_score']
neighbors = list(grid3.cv_results_['param_n_neighbors'])


# In[44]:


plt.plot(neighbors, score1)
plt.plot(neighbors, score2)
plt.plot(neighbors, score3)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# # Regression tree

# In[48]:


df_new['language'] = df_new['language'].apply(lambda x: 1 if x==1 else 0)


# In[51]:


train_1, test_1 = train_test_split(df_new, test_size = 0.3, random_state=0)
x_train_1 = train_1.drop(['popularity','pop'], axis=1)
y_train_1 = train_1['popularity']
x_test_1 = test_1.drop(['popularity','pop'], axis=1)
y_test_1 = test_1['popularity']


# In[52]:


tree_1 = tree.DecisionTreeRegressor(max_depth=3)
tree_1.fit(x_train_1,y_train_1)

print("Simple Tree Training R-Square",tree_1.score(x_train_1,y_train_1))
print("Simple Tree Testing R-Square",tree_1.score(x_test_1,y_test_1))

rf_1 = RandomForestRegressor(n_jobs=8)
rf_1.fit(x_train_1,y_train_1)

print("RandomForest Training R-Square",rf_1.score(x_train_1,y_train_1))
print("RandomForest Testing R-Square",rf_1.score(x_test_1,y_test_1))

xgb_rb_1 = xgb.XGBRegressor()
xgb_rb_1.fit(x_train_1, y_train_1)

print("Boosted Tree Training R-Square",xgb_rb_1.score(x_train_1,y_train_1))
print("Boosted Tree Testing R-Square",xgb_rb_1.score(x_test_1,y_test_1))


# In[53]:


def calc_rmse(pred,actual):
    return np.sqrt(np.sum((pred - actual)**2)/len(pred))


# In[54]:


tree_pred_train1 = tree_1.predict(x_train_1)
xgb_pred_train1 = xgb_rb_1.predict(x_train_1)
rf_pred_train1 = rf_1.predict(x_train_1)
print(calc_rmse(tree_pred_train1, y_train_1),calc_rmse(rf_pred_train1, y_train_1),calc_rmse(xgb_pred_train1, y_train_1))


# In[55]:


tree_pred_test1 = tree_1.predict(x_test_1)
xgb_pred_test1 = xgb_rb_1.predict(x_test_1)
rf_pred_test1 = rf_1.predict(x_test_1)
print(calc_rmse(tree_pred_test1, y_test_1),calc_rmse(rf_pred_test1, y_test_1),calc_rmse(xgb_pred_test1, y_test_1))


# In[56]:


df_2 = df_new[['year','key','mode','tempo','#vocab','avg_word_len','uniqueness','language','comm','negative','anticipation',
               'anger','disgust','fear','joy','positive','sadness','surprise','trust','duration','popularity']]
train_2, test_2 = train_test_split(df_2, test_size = 0.3, random_state=0)
x_train_2 = train_2.drop(['popularity'], axis=1)
y_train_2 = train_2['popularity']
x_test_2 = test_2.drop(['popularity'], axis=1)
y_test_2 = test_2['popularity']


# In[57]:


tree_2 = tree.DecisionTreeRegressor(max_depth=3)
tree_2.fit(x_train_2,y_train_2)

#Get the R-Square for the predicted vs actuals on the test sample
print("Simple Tree Training R-Square",tree_2.score(x_train_2,y_train_2))
print("Simple Tree Testing R-Square",tree_2.score(x_test_2,y_test_2))

rf_2 = RandomForestRegressor(n_jobs=8)
rf_2.fit(x_train_2,y_train_2)

print("Random Forest Training R-Square",rf_2.score(x_train_2,y_train_2))
print("Random Forest Testing R-Square",rf_2.score(x_test_2,y_test_2))

xgb_rb_2 = xgb.XGBRegressor()
xgb_rb_2.fit(x_train_2, y_train_2)

print("Boosted Tree Training R-Square",xgb_rb_2.score(x_train_2,y_train_2))
print("Boosted Tree Testing R-Square",xgb_rb_2.score(x_test_2,y_test_2))


# In[58]:


tree_pred_train2 = tree_2.predict(x_train_2)
xgb_pred_train2 = xgb_rb_2.predict(x_train_2)
rf_pred_train2 = rf_2.predict(x_train_2)
print(calc_rmse(tree_pred_train2, y_train_2),calc_rmse(rf_pred_train2, y_train_2),calc_rmse(xgb_pred_train2, y_train_2))


# In[59]:


tree_pred_test2 = tree_2.predict(x_test_2)
xgb_pred_test2 = xgb_rb_2.predict(x_test_2)
rf_pred_test2 = rf_2.predict(x_test_2)
print(calc_rmse(tree_pred_test2, y_test_2),calc_rmse(rf_pred_test2, y_test_2),calc_rmse(xgb_pred_test2, y_test_2))


# In[60]:


df_3 = df_new[['year','key','mode','tempo','duration', 'popularity']]
train_3, test_3 = train_test_split(df_3, test_size = 0.3, random_state=0)
x_train_3 = train_3.drop(['popularity'], axis=1)
y_train_3 = train_3['popularity']
x_test_3 = test_3.drop(['popularity'], axis=1)
y_test_3 = test_3['popularity']


# In[61]:


tree_3 = tree.DecisionTreeRegressor(max_depth=3)
tree_3.fit(x_train_3,y_train_3)

print("Simple Tree Training R-Square",tree_3.score(x_train_3,y_train_3))
print("Simple Tree Testing R-Square",tree_3.score(x_test_3,y_test_3))

rf_3 = RandomForestRegressor(n_jobs=8)
rf_3.fit(x_train_3,y_train_3)

print("Random Forest Training R-Square",rf_3.score(x_train_3,y_train_3))
print("Random Forest Testing R-Square",rf_3.score(x_test_3,y_test_3))


xgb_rb_3 = xgb.XGBRegressor()
xgb_rb_3.fit(x_train_3, y_train_3)

print("Boosted Tree Training R-Square",xgb_rb_3.score(x_train_3,y_train_3))
print("Boosted Tree Testing R-Square",xgb_rb_3.score(x_test_3,y_test_3))


# In[62]:


tree_pred_train3 = tree_3.predict(x_train_3)
xgb_pred_train3 = xgb_rb_3.predict(x_train_3)
rf_pred_train3 = rf_3.predict(x_train_3)
print(calc_rmse(tree_pred_train3, y_train_3),calc_rmse(rf_pred_train3, y_train_3),calc_rmse(xgb_pred_train3, y_train_3))


# In[63]:


tree_pred_test3 = tree_3.predict(x_test_3)
xgb_pred_test3 = xgb_rb_3.predict(x_test_3)
rf_pred_test3 = rf_3.predict(x_test_3)
print(calc_rmse(tree_pred_test3, y_test_3),calc_rmse(rf_pred_test3, y_test_3),calc_rmse(xgb_pred_test3, y_test_3))


# In[64]:


param_grid_reg = dict(
    max_depth = [4, 5, 6, 7],
    learning_rate = np.linspace(0.03, 0.3, 5),
    n_estimators = [100, 200, 300]
)


# In[65]:


grid_reg = GridSearchCV(xgb_rb_1,param_grid_reg,cv=5, n_jobs=8,scoring = 'neg_mean_squared_error')
grid_reg.fit(x_train_1, y_train_1) 


# In[66]:


xgb_1_new_reg = grid_reg.best_estimator_
xgb_1_new_reg.fit(x_train_1, y_train_1)

print("Boosted Tree Training R-Square",xgb_1_new_reg.score(x_train_1,y_train_1))
print("Boosted Tree Testing R-Square",xgb_1_new_reg.score(x_test_1,y_test_1))


# In[67]:


print("previous xgboost model rmse: ", calc_rmse(xgb_rb_1.predict(x_test_1), y_test_1))
print("xgboost rmse after hypertuning:", calc_rmse(xgb_1_new_reg.predict(x_test_1), y_test_1))


# In[68]:


import shap
shap.initjs() 
explainer = shap.TreeExplainer(xgb_rb_1)
shap_values = explainer.shap_values(x_train_1)
#for i in range(10):
#    shap.force_plot(explainer.expected_value, shap_values[i,:], x_train_1.iloc[i,:])


# In[69]:


shap.summary_plot(shap_values, x_train_1)


# In[ ]:




