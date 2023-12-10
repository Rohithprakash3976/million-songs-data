#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import Recommenders as Recommenders


# In[7]:


song_df_1 = pd.read_csv('triplets_file.csv')
song_df_1.head()


# In[8]:


song_df_2 = pd.read_csv('song_data.csv')
song_df_2.head()


# In[9]:


song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on='song_id', how='left')
song_df.head()


# In[10]:


len(song_df)


# In[11]:


print(len(song_df_1), len(song_df_2))


# In[29]:


song_df['song'] = song_df['title']+' - '+song_df['artist_name']
song_df.head()


# In[13]:


##cummilative sum of listen count of the songs
song_grouped = song_df.groupby(['song']).agg({'listen_count':'count'}).reset_index()
song_grouped.head()


# In[14]:


grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage'] = (song_grouped['listen_count'] / grouped_sum ) * 100
song_grouped.sort_values(['listen_count', 'song'], ascending=[0,1])


# In[15]:


##Popularity Recomandation Engine


# In[16]:


pr = Recommenders.popularity_recommender_py()


# In[17]:


pr.create(song_df, 'user_id', 'song')


# In[18]:


pr.recommend(song_df['user_id'][5])


# In[19]:


##Item Similarity Recommendation


# In[20]:


ir = Recommenders.item_similarity_recommender_py()
ir.create(song_df, 'user_id', 'song')


# In[23]:


user_items = ir.get_user_items(song_df['user_id'][10])


# In[24]:


for user_item in user_items:
    print(user_item)


# In[26]:


ir.recommend(song_df['user_id'][10])


# In[27]:


ir.get_similar_items(['U Smile-Justin Bieber'])


# In[ ]:




