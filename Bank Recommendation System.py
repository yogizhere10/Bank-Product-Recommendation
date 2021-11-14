#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[136]:


df = pd.read_csv('Recommendation_train.csv')


# In[137]:


df.head()


# In[138]:


unwanted = ['[', ']']


# In[139]:


for i in range(df.shape[0]):
    for j in unwanted:
        df['Product_Holding_B12'][i] = df['Product_Holding_B12'][i].replace(j, '') 


# In[140]:


df.head()


# In[141]:


df.nunique()


# In[142]:


df.shape


# In[143]:


df.isnull().sum()


# In[144]:


object = [feature for feature in df.columns if df[feature].dtypes == 'O' and feature not in ['Customer_ID', 'Product_Holding_B2', 'Product_Holding_B12']]


# In[145]:


object


# In[146]:


from sklearn.preprocessing import OneHotEncoder


# In[147]:


ohe = OneHotEncoder()


# In[148]:


final_encoded = []
for value in object:
    encoded = ohe.fit_transform(df[value].values.reshape(-1,1)).toarray()
    n = df[value].nunique()
    cols = ['{}_{}'.format(value, n) for n in range(1, n+1)]
    encoded_df = pd.DataFrame(encoded, columns = cols)
    encoded_df.index = df.index
    final_encoded.append(encoded_df)

df_final = pd.concat([df, *final_encoded], axis = 1)


# In[149]:


df_final.head()


# In[150]:


df_final.drop(['Gender_1', 'City_Category_1', 'Customer_Category_1'], axis = 1, inplace = True)


# In[151]:


df_final.head()


# In[152]:


df_final.drop(object,axis = 1, inplace = True)


# In[153]:


df1 = df_final.groupby(by = ['Product_Holding_B12'])['Customer_ID'].count().reset_index().rename(columns = {'Customer_ID':'Total'})


# In[154]:


df1.head()


# In[155]:


df_merged = df_final.merge(df1, left_on = 'Product_Holding_B12', right_on = 'Product_Holding_B12', how = 'left')


# In[158]:


df_merged.head()


# In[157]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
df_merged['Total'] = sc.fit_transform(df_merged[['Total']])


# In[159]:


df_pivot = df_merged.pivot(index = 'Product_Holding_B12', columns = 'Customer_ID', values = 'Total').fillna(0)


# In[ ]:





# In[160]:


from scipy.sparse import csr_matrix
df_pivot_matrix = csr_matrix(df_pivot.values)


# In[161]:


df_pivot.head(15)


# In[162]:


from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')


# In[202]:


model_knn.fit(df_pivot_matrix)


# In[203]:


query = np.random.choice(df_pivot.shape[0])
print(query)


# In[204]:


df_pivot.index[query]


# In[205]:


distances, indices = model_knn.kneighbors(df_pivot.iloc[query,:].values.reshape(1,-1), n_neighbors = 6)


# In[206]:



for i in range(len(distances.flatten())):
    if i == 0:
        print('Recommendation for {0}:\n'.format(df_pivot.index[query]))
    else:
        print('{0}: {1} with distance of {2}'.format(i, df_pivot.index[indices.flatten()[i]], distances.flatten()[i]))


# In[207]:


indices


# In[208]:


np.max(indices)


# In[210]:


df_pivot.index[np.max(indices)]


# In[178]:


from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


# In[187]:


item_similarity = cosine_similarity(df_pivot)


# In[188]:


item_similarity[0]


# In[ ]:





# In[ ]:




