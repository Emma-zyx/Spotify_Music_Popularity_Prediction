#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
from sklearn import tree


# In[4]:


df=pd.read_csv('data.csv')
df.info()


# In[5]:


df_artist=pd.read_csv('data_by_artist.csv')
df_artist.info()


# In[6]:


df.describe()


# In[7]:


df['duration_min'] = df['duration_ms']/60000


# In[8]:


df.drop(['duration_ms','id'],axis=1,inplace=True)


# In[9]:


df.dropna(inplace=True)
df['pop'] = np.where(df['popularity']<33, 0, 1)
df.describe()


# In[10]:


df_genre = pd.read_csv("data_by_genres.csv")
df_genre.describe()


# In[11]:


df_genre['duration_min'] = df_genre['duration_ms']/60000


# In[12]:


df_genre.drop(['duration_ms'],axis=1,inplace=True)


# In[13]:


df_genre.dropna(inplace=True)
df_genre.describe()


# In[14]:


columns = ["acousticness","danceability","energy","speechiness","liveness","valence"]
for col in columns:
    x = df.groupby("year")[col].mean()
    sns.set(style='dark',)
    sns.set(rc={'figure.figsize':(15,8)})
    ax= sns.lineplot(x=x.index,y=x,label=col)
ax.set_title('Audio characteristics over year')
ax.set_ylabel('Measure')
ax.set_xlabel('Year')


# In[15]:


df_artist = pd.read_csv("data_by_artist.csv")


# In[16]:


plt.figure(figsize=(16, 4))
sns.set(style="whitegrid")
x = df.groupby("artists")["popularity"].sum().sort_values(ascending=False).head(20)
ax = sns.barplot(x.index, x,palette="Blues_d")
ax.set_title('Top Artists with Popularity',y=1.1,fontsize=20)
ax.set_ylabel('Popularity')
ax.set_xlabel('Artists')
plt.xticks(rotation = 45)


# In[17]:


#sns.set(style="whitegrid")
x = df_artist.groupby("artists")["popularity"].sum().sort_values(ascending=False).head(15)
fig = plt.figure(figsize=(16, 4))
ax = sns.barplot(x.index, x,palette="Blues_d")
plt.title('Top Artists with Popularity',y=1.1,fontsize=20)
plt.ylabel('Popularity')
plt.xlabel('Artists')
plt.xticks(rotation = 45)
ax.axes.get_xaxis().set_visible(True)


# In[18]:


pop = df.groupby("name")["popularity"].sum().sort_values(ascending=False).head(15)
fig = plt.figure(figsize=(16,4))
ax = sns.barplot(pop.index,pop,palette="Blues_d")
plt.xticks(rotation=45,ha='right')
plt.title('Top 10 Most Popular Songs from 1921-2020',y=1.1,fontsize=20)
plt.xlabel('Songs')
plt.ylabel('Popularity (Ranges from 0 to 100)')
ax.axes.get_xaxis().set_visible(True)


# In[19]:


year = pd.DataFrame(df['year'].value_counts())
year = year.sort_index()
ax=year.plot(kind='line',figsize=(16,4), linewidth=2)
plt.title("Number of songs released each year")
plt.xlabel('Year')
plt.ylabel('Number of songs')
ax.axes.get_xaxis().set_visible(True)


# In[20]:


mode = pd.DataFrame(df['mode'].value_counts())
mode['mode'].plot(kind='pie',
            figsize=(15, 6),
            autopct='%1.1f%%', 
            startangle=90)
plt.title('Mode Ratio',y=1.1,fontsize=25) 
plt.axis('equal') 
plt.legend(labels=['Major','Minor'], loc='upper left') 
plt.show()


# In[21]:


pop = pd.DataFrame(df['pop'].value_counts())
pop['pop'].plot(kind='pie',
            figsize=(15, 6),
            autopct='%1.1f%%', 
            startangle=90)
plt.title('Popular Ratio',y=1.1,fontsize=25) 
plt.axis('equal') 
plt.legend(labels=['Popular','Not as popular'], loc='upper left') 
plt.show()


# In[22]:


ex = pd.DataFrame(df['explicit'].value_counts())
ex['explicit'].plot(kind='pie',
            figsize=(15, 6),
            autopct='%1.1f%%', 
            startangle=90)
plt.title('Explicit Ratio',y=1.1,fontsize=25) 
plt.axis('equal') 
plt.legend(labels=['Not Explicit','Explicit'], loc='upper left') 
plt.show()


# In[23]:


key = pd.DataFrame(df['key'].value_counts())
key['key'].plot(kind='pie',
            figsize=(15, 6),
            autopct='%1.1f%%', 
            startangle=90)
plt.title('Key Ratio',y=1.1,fontsize=25) 
plt.axis('equal') 
#plt.legend(labels=['Not Explicit','Explicit'], loc='upper left') 
plt.show()


# In[24]:


df['dance'] = np.where(df['danceability']<0.5, 0, 1)
ax = sns.boxplot(x="dance", y="popularity", data=df)


# In[25]:


df['acoustic'] = np.where(df['acousticness']<0.5, 0, 1)
ax = sns.boxplot(x="acoustic", y="popularity", data=df)


# In[26]:


df['speech'] = np.where(df['speechiness']<0.5, 0, 1)
ax = sns.boxplot(x="speech", y="popularity", data=df)


# In[32]:


df['live'] = np.where(df['liveness']<0.5, 0, 1)
ax = sns.boxplot(x="live", y="popularity", data=df)


# In[28]:


df['energy_new'] = np.where(df['energy']<0.5, 0, 1)
ax = sns.boxplot(x="energy_new", y="popularity", data=df)


# In[29]:


df['loud'] = np.where(df['loudness']<0.5, 0, 1)
ax = sns.boxplot(x="loud", y="popularity", data=df)


# In[30]:


#df['explicit_'] = np.where(df['explicit']<0.5, 0, 1)
ax = sns.boxplot(x="explicit", y="popularity", data=df)


# In[21]:


key = pd.DataFrame(df['key'].value_counts()).reset_index().sort_values('index')
key.replace({'index' : { 0 : 'C', 1 : 'C#', 2 : 'D', 3 : 'D#', 4 : 'E', 5 : 'F', 6 : 'F#', 
                        7 : 'G', 8 : 'G#', 9 : 'A', 10 : 'A#', 11 : 'B'}} , inplace=True)
fig = plt.figure(figsize=(15,6))
ax = sns.barplot(key['index'],key['key'],palette="Blues_d")
plt.title('Frequency Count For Key',y=1.1,fontsize=20)
plt.xlabel('Key')
plt.ylabel('Frequency')
ax.axes.get_xaxis().set_visible(True)


# In[22]:


keypop = pd.DataFrame(df.groupby('key')['popularity'].mean()).reset_index()
keypop.replace({'key' : { 0 : 'C', 1 : 'C#', 2 : 'D', 3 : 'D#', 4 : 'E', 5 : 'F', 6 : 'F#', 
                        7 : 'G', 8 : 'G#', 9 : 'A', 10 : 'A#', 11 : 'B'}} , inplace=True)

fig = plt.figure(figsize=(15,6))
ax = sns.barplot(keypop['key'], keypop['popularity'],palette="Blues_d")
plt.title('Key VS Popularity',y=1.1,fontsize=20)
plt.xlabel('Key')
plt.ylabel('Popularity')
ax.axes.get_xaxis().set_visible(True)


# In[23]:


x = df_genre.groupby("genres")["popularity"].sum().sort_values(ascending=False).head(15)
fig = plt.figure(figsize=(15,4))
ax = sns.barplot(x.index, x,palette="Blues_d")
plt.title('Genres VS Popularity',y=1.1,fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Popularity',fontsize=20)
plt.xticks(fontsize=20,rotation=45,ha='right')
ax.axes.get_xaxis().set_visible(True)


# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')
COL_NUM = 2
ROW_NUM = 5
import matplotlib.pyplot as plt
fig, axes = plt.subplots(ROW_NUM, COL_NUM, figsize=(15,15))

features = ['acousticness','danceability','energy','instrumentalness','valence','liveness','speechiness','loudness','tempo']
for i in features:
    ax = axes[int(features.index(i)/COL_NUM), features.index(i)%COL_NUM]
    feature= df.groupby('year')[i].mean().reset_index()
    ax.plot('year',i,data=feature,linewidth=1.3)
    ax.set_title(i)
    ax.set_xlabel('Years')
plt.tight_layout() 


# In[25]:


df_genre.corr()


# In[26]:


plt.pcolor(df_genre.corr(),cmap='coolwarm') #https://matplotlib.org/examples/color/colormaps_reference.html
plt.show()


# In[27]:


df_genre2=pd.read_csv("data_w_genres.csv")


# In[28]:


df_genre2.describe()


# In[29]:


index_name=df_genre2[df_genre2['genres']=="[]"].index
df_genre2.drop(index_name,inplace=True)


# In[30]:


df_genre2['duration_min'] = df_genre2['duration_ms']/60000
df_genre2.drop(['duration_ms'],axis=1,inplace=True)


# In[31]:


df_genre2.corr()


# In[32]:


plt.pcolor(df_genre2.corr(),cmap='coolwarm') #https://matplotlib.org/examples/color/colormaps_reference.html
plt.show()


# In[33]:


len(df_genre['key'].unique())


# In[34]:


#top genres for each key
key_genre = df_genre2.groupby(['genres','key']).size().unstack()
get_ipython().run_line_magic('matplotlib', 'inline')
COL_NUM = 2
ROW_NUM = 6
import matplotlib.pyplot as plt
fig, axes = plt.subplots(ROW_NUM, COL_NUM, figsize=(15,15))

for i, (key,genre_count) in enumerate(key_genre.items()): 
    ax = axes[int(i/COL_NUM), i%COL_NUM]
    genre_count = genre_count.sort_values(ascending=False)[:5] 
    genre_count.plot(kind='barh',ax=ax)
    ax.set_title(key)

plt.tight_layout() 


# In[36]:


#key and features
get_ipython().run_line_magic('matplotlib', 'inline')
COL_NUM = 2
ROW_NUM = 5
import matplotlib.pyplot as plt
fig, axes = plt.subplots(ROW_NUM, COL_NUM, figsize=(15,15))

features = ['acousticness','danceability','energy','instrumentalness','valence','liveness','speechiness','loudness','tempo']
for i in features:
    ax = axes[int(features.index(i)/COL_NUM), features.index(i)%COL_NUM]
    genre= df_genre.groupby('key')[i].mean().reset_index()
    ax.plot('key',i,data=genre,linewidth=1.3)
    ax.set_title(i)
    ax.set_xlabel('Years')
plt.tight_layout() 


# In[37]:


df['decade'] = (df['year']//10)*10
df['popularity_range'] = (df['popularity']//10)*10
df[df['popularity_range'] == 100]['popularity_range'] = 90
decade = pd.DataFrame(df['decade'].value_counts()).reset_index().sort_values('index')
fig = plt.figure(figsize=(15,6))
ax = sns.barplot(decade['index'], decade['decade'],palette="Blues_d")
plt.title('Number of Songs by Decade',y=1.1,fontsize=20)
plt.xlabel('Decade',fontsize=20)
plt.ylabel('Frequency',fontsize=20)
plt.xticks(fontsize=20,rotation=45,ha='right')
ax.axes.get_xaxis().set_visible(True)


# In[38]:


x = df.groupby("decade")["energy"].mean()
fig = plt.figure(figsize=(15, 6))
ax = sns.barplot(x.index, x,palette="Blues_d")
plt.title('Average energy by decade',y=1.1,fontsize=20)
plt.ylabel('energy')
plt.xlabel('decade')
plt.xticks(rotation = 45)
ax.axes.get_xaxis().set_visible(True)


# In[39]:


x = df.groupby("decade")["danceability"].mean()
fig = plt.figure(figsize=(15, 6))
ax = sns.barplot(x.index, x,palette="Blues_d")
plt.title('Average danceability by decade',y=1.1,fontsize=20)
plt.ylabel('danceability')
plt.xlabel('decade')
plt.xticks(rotation = 45)
ax.axes.get_xaxis().set_visible(True)


# In[40]:


x = df.groupby("decade")["instrumentalness"].mean()
fig = plt.figure(figsize=(15, 6))
ax = sns.barplot(x.index, x,palette="Blues_d")
plt.title('Average instrumentalness by decade',y=1.1,fontsize=20)
plt.ylabel('instrumentalness')
plt.xlabel('decade')
plt.xticks(rotation = 45)
ax.axes.get_xaxis().set_visible(True)


# In[41]:


x = df.groupby("decade")["acousticness"].mean()
fig = plt.figure(figsize=(15, 6))
ax = sns.barplot(x.index, x,palette="Blues_d")
plt.title('Average acousticness by decade',y=1.1,fontsize=20)
plt.ylabel('acousticness')
plt.xlabel('decade')
plt.xticks(rotation = 45)
ax.axes.get_xaxis().set_visible(True)


# In[42]:


x = df.groupby("decade")["popularity"].mean()
fig = plt.figure(figsize=(15, 6))
ax = sns.barplot(x.index, x,palette="Blues_d")
plt.title('Average Popularity by Decade',y=1.1,fontsize=20)
plt.ylabel('popularity')
plt.xlabel('decade')
plt.xticks(rotation = 45)
ax.axes.get_xaxis().set_visible(True)


# In[43]:


df.popularity_range.replace([0,10,20,30,40,50,60,70,80,90,100], ['0-10','10-20','20-30','30-40','40-50','50-60','60-70',
                                                            '70-80','80-90','90-100','90-100'], inplace=True)


# In[44]:


pop_range = pd.DataFrame(df['popularity_range'].value_counts()).reset_index().sort_values('index')

fig = plt.figure(figsize=(15,6))
ax = sns.barplot(pop_range['index'], pop_range['popularity_range'],palette="Blues_d")
plt.title('Number of Songs within each Popularity Range',y=1.1,fontsize=20)
plt.xlabel('Popularity Range',fontsize=20)
plt.ylabel('Frequency',fontsize=20)
plt.xticks(fontsize=20,rotation=45,ha='right')
ax.axes.get_xaxis().set_visible(True)


# In[45]:


x = df.groupby('popularity_range')["danceability"].mean()
fig = plt.figure(figsize=(15, 6))
ax = sns.barplot(x.index, x,palette="Blues_d")
plt.title('Average Danceability within each Popularity Range',y=1.1,fontsize=20)
plt.ylabel('danceability')
plt.xlabel('popularity range')
plt.xticks(rotation = 45)
ax.axes.get_xaxis().set_visible(True)


# In[46]:


x = df.groupby('popularity_range')["instrumentalness"].mean()
fig = plt.figure(figsize=(15, 6))
ax = sns.barplot(x.index, x,palette="Blues_d")
plt.title('Average Instrumentalness within each Popularity Range',y=1.1,fontsize=20)
plt.ylabel('instrumentalness')
plt.xlabel('popularity range')
plt.xticks(rotation = 45)
ax.axes.get_xaxis().set_visible(True)


# In[47]:


x = df.groupby('popularity_range')["acousticness"].mean()
fig = plt.figure(figsize=(15, 6))
ax = sns.barplot(x.index, x,palette="Blues_d")
plt.title('Average Acousticness within each Popularity Range',y=1.1,fontsize=20)
plt.ylabel('acousticness')
plt.xlabel('popularity range')
plt.xticks(rotation = 45)
ax.axes.get_xaxis().set_visible(True)


# In[48]:


test_df = df[['valence', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']]


# In[49]:


plt.figure(figsize=(16, 16))
t = 1
for i in test_df.columns:
    plt.subplot(3, 3, t)
    plt.title(i)
    df[i].plot(kind="hist", alpha=0.8)
    t+=1


# In[56]:


#plt.figure(figsize = (15, 13))
corr = df.corr()
fix, ax = plt.subplots(figsize=(50,30))
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
ax = sns.heatmap(corr, mask=mask, ax=ax, annot= True, cmap='bwr')
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=20)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=17)
plt.savefig('heatmap', dpi=300)
plt.show()

#sns.heatmap(corr, annot = True, cmap='coolwarm',linewidths=.1)

