#!/usr/bin/env python
# coding: utf-8

# In[1]:


# preserved line
# Date: 2021-10-07
# this notebook is compared the results for the raw data (before remove and clean the data)


# In[2]:


# import the library
import pandas as pd


# In[3]:


# define some functions
def myShowDf(x):
    return(x[:10])


# In[4]:


fileOverviewDf = pd.read_csv('./course/COMP5331/resFileListPwd_clean2.csv', index_col=0)


# In[5]:


fileOverviewDf


# In[13]:


metaDataDf = pd.read_csv('./course/COMP5331/Coswara-Data/combined_data_clean2.csv')


# In[14]:


metaDataDf


# In[15]:


# is the number of the patients the same with the numebr of the meta data?
fileOverviewDf.shape[0] == metaDataDf.shape[0]


# In[16]:


# if true, we just combine them by the index
metaDataDf_indexed = metaDataDf.set_index(metaDataDf['id'])
fileOverviewDf_indexed = fileOverviewDf.set_index(fileOverviewDf['patientNam'])
# metaDataDf.sort_values(['id'], ascending=True)
# fileOverviewDf.sort_values(['patientNam'], ascending=True)


# In[17]:


# concatenate the indexed df
concat_fileOverviewDf = pd.concat([fileOverviewDf_indexed, metaDataDf_indexed], axis=1)
concat_fileOverviewDf.set_index(pd.Index(list(range(concat_fileOverviewDf.shape[0]))),
                               inplace=True) #inplace=True, not copy and just change


# In[18]:


myShowDf(concat_fileOverviewDf)


# In[19]:


myShowDf(concat_fileOverviewDf.sort_values(['fileNum'], ascending=True))


# In[20]:


myShowDf(concat_fileOverviewDf.sort_values(['fileNum'], ascending=False))


# In[ ]:





# In[ ]:





# # Following codes are used for dropping two patients in raw data

# In[23]:


# metaDataDf_clean = metaDataDf.drop(metaDataDf[metaDataDf['id'] == 'o0HUIrKBsMXkygZgI161LVIydIp1'].index)
# metaDataDf_clean.shape


# In[24]:


# metaDataDf_clean = metaDataDf_clean.drop(metaDataDf_clean[metaDataDf_clean['id'] == '9hftEYixyhP1Neeq3fB7ZwITQC53'].index)
# metaDataDf_clean.shape


# In[11]:


# use the loop to drop the data
import os
dropList = os.listdir('./course/COMP5331/removeCoswara/audioFolder/audioSizeZero')
metaDataDf_clean2 = metaDataDf
for i in dropList:
    metaDataDf_clean2 = metaDataDf_clean2.drop(metaDataDf_clean2[metaDataDf_clean2['id'] == i].index)

metaDataDf_clean2.shape


# In[12]:


# # write the clean meta data into csv
metaDataDf_clean2.to_csv('./course/COMP5331/Coswara-Data/combined_data_clean2.csv')


# In[ ]:
