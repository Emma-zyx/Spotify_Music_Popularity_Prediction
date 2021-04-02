#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
df = pd.read_csv('data.csv',na_values='null')


# In[7]:


import re
df['name'] = df['name'].apply(lambda x: x.lower())
df['name'] = df['name'].apply(lambda x: re.sub("[^a-zA-Z]+", " ", x))


# ## Emotion

# In[24]:


def get_nrc_data():
    nrc = "data/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
    count=0
    emotion_dict=dict()
    with open(nrc,'r') as f:
        all_lines = list()
        for line in f:
            if count < 46:
                count+=1
                continue
            line = line.strip().split('\t')
            if int(line[2]) == 1:
                if emotion_dict.get(line[0]):
                    emotion_dict[line[0]].append(line[1])
                else:
                    emotion_dict[line[0]] = [line[1]]
    return emotion_dict


# In[25]:


emotion_dict = get_nrc_data()


# In[26]:


def emotion_analyzer(text,emotion_dict=emotion_dict):
    #Set up the result dictionary
    emotions = {x for y in emotion_dict.values() for x in y}
    emotion_count = dict()
    for emotion in emotions:
        emotion_count[emotion] = 0

    #Analyze the text and normalize by total number of words
    total_words = len(text.split())
    for word in text.split():
        if emotion_dict.get(word):
            for emotion in emotion_dict.get(word):
                emotion_count[emotion] += 1/len(text.split())
    return emotion_count

df['emotion'] = df['name'].apply(lambda x: emotion_analyzer(x))


# In[27]:


def negative(dic):
    return dic.get('negative')

df['negative'] = df['emotion'].apply(lambda x: negative(x))


# In[28]:


def anticipation(dic):
    return dic.get('anticipation')

df['anticipation'] = df['emotion'].apply(lambda x: anticipation(x))


# In[29]:


def anger(dic):
    return dic.get('anger')

df['anger'] = df['emotion'].apply(lambda x: anger(x))


# In[30]:


def disgust(dic):
    return dic.get('disgust')

df['disgust'] = df['emotion'].apply(lambda x: disgust(x))


# In[31]:


def fear(dic):
    return dic.get('fear')

df['fear'] = df['emotion'].apply(lambda x: fear(x))


# In[32]:


def joy(dic):
    return dic.get('joy')

df['joy'] = df['emotion'].apply(lambda x: joy(x))


# In[33]:


def positive(dic):
    return dic.get('positive')

df['positive'] = df['emotion'].apply(lambda x: positive(x))


# In[34]:


def sadness(dic):
    return dic.get('sadness')

df['sadness'] = df['emotion'].apply(lambda x: sadness(x))


# In[35]:


def surprise(dic):
    return dic.get('surprise')

df['surprise'] = df['emotion'].apply(lambda x: surprise(x))


# In[36]:


def trust(dic):
    return dic.get('trust')

df['trust'] = df['emotion'].apply(lambda x: trust(x))


# In[37]:


df


# In[38]:


df.describe()


# In[40]:


df.to_csv('data_updated.csv')


# In[ ]:




