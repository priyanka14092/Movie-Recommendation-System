#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


columns_name=['user_id','item_id','rating','timestamp']
df=pd.read_csv('u.data',sep="\t",names=columns_name)


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df['user_id']


# In[7]:


df['user_id'].nunique()


# In[8]:


df['item_id'].nunique()


# In[9]:


movies_title=pd.read_csv('u.item',sep="\|",header=None)


# In[10]:


movies_title.shape


# In[11]:


movies_titles=movies_title[[0,1]]
movies_titles.columns=["item_id","title"]
movies_titles.head()


# In[12]:


df=pd.merge(df,movies_titles,on="item_id")


# In[13]:


df


# In[14]:


df.tail()


# In[15]:


ratings=pd.DataFrame(df.groupby('title').mean()['rating'])


# In[16]:


ratings.head()


# In[17]:


ratings['num of ratings']=pd.DataFrame(df.groupby('title').count()['rating'])


# #Create the recommender system

# In[18]:


df.head()


# In[19]:


moviemat=df.pivot_table(index="user_id",columns="title",values="rating")


# In[20]:


moviemat.head()


# In[21]:


starwars_user_ratings=moviemat['Star Wars (1977)']


# In[22]:


starwars_user_ratings.head(20)


# In[23]:


similar_to_starwars=moviemat.corrwith(starwars_user_ratings)


# In[24]:


similar_to_starwars


# In[25]:


corr_starwars=pd.DataFrame(similar_to_starwars,columns=['correlation'])


# In[26]:


corr_starwars.dropna(inplace=True)


# In[27]:


corr_starwars


# In[28]:


corr_starwars.head()


# In[29]:


corr_starwars.sort_values('correlation',ascending=False).head(10)


# In[30]:


ratings


# In[31]:


corr_starwars=corr_starwars.join(ratings['num of ratings'])


# In[32]:


corr_starwars


# In[33]:


corr_starwars.head()


# In[34]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('correlation',ascending=False)


# In[38]:


def predict_movies(movie_name):
    movie_user_ratings=moviemat[movie_name]
    similar_to_movie=moviemat.corrwith(movie_user_ratings)
    corr_movie=pd.DataFrame(similar_to_movie,columns=['correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(ratings['num of ratings'])
    
    predictions=corr_movie[corr_movie['num of ratings']>100].sort_values('correlation',ascending=False)
    
    return predictions


# In[39]:


predict_my_movie=predict_movies("Titanic (1997)")


# In[40]:


predict_my_movie.head()

