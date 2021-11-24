#!/usr/bin/env python
# coding: utf-8

# # This is the notebook for read the files after cleanning

# In[3]:


# input the pwd
import os
# folderPwd = '/home/jligm/share80T/course/COMP5331/Coswara-Data/20200413/20200413'


# In[4]:


# define some function to show the data
def myShowList(x):
    print(x[:10])


# In[ ]:





# In[5]:


# subfolderList = os.listdir(folderPwd)


# In[6]:


# myShowList(subfolderList)


# In[7]:


# for ssFolder in subfolderList:
#     ssPwd = os.path.join(folderPwd, ssFolder)

#     print(os.listdir(ssPwd))


# In[8]:


# ssFolder


# In[9]:


# folderPwd = '/home/jligm/share80T/course/COMP5331/Coswara-Data/20200413/20200413'
# subfolderList = os.listdir(folderPwd)
# for ssFolder in subfolderList:
#     ssPwd = os.path.join(folderPwd, ssFolder)
#     print(os.listdir(ssPwd))


# In[10]:


allFolderList = [i for i in os.listdir('./course/COMP5331/Coswara-Data/') if '20' in i]
myShowList(allFolderList)


# In[11]:


# ssFolder


# In[12]:


# # subfolderList
# len(os.listdir(ssPwd))


# In[13]:


fullList = ['breathing-shallow.wav', 'counting-fast.wav', 'cough-heavy.wav', 'vowel-e.wav', 'cough-shallow.wav', 'vowel-o.wav', 'metadata.json', 'vowel-a.wav', 'counting-normal.wav', 'breathing-deep.wav']


# In[14]:


testList = ['counting-fast.wav', 'cough-heavy.wav', 'vowel-e.wav', 'cough-shallow.wav', 'vowel-o.wav', 'metadata.json', 'vowel-a.wav', 'counting-normal.wav', 'breathing-deep.wav']


# In[15]:


[i for i in fullList if i not in testList]


# In[16]:


[i for i in testList if i not in fullList]


# In[17]:


# create the result list
dateRes = []
patientRes = []
fileRes = []
fileNumRes = []
fileNotHave = []
fileAddition = []

for j in allFolderList:
    pwd = os.path.join('./course/COMP5331/Coswara-Data/',j,j)
    # list all the files in the folder
#     print(pwd)
    subfolderList = os.listdir(pwd)
    for ssFolder in subfolderList:
        if 'csv' in ssFolder or 'DS_Store' in ssFolder:
            continue
        ssPwd = os.path.join(pwd, ssFolder)
#         test = os.listdir(ssPwd)
        tmpFile = os.listdir(ssPwd)
        tmpFileNum = len(tmpFile)

        testList = tmpFile
        tmpNotHave = [i for i in fullList if i not in testList]
        tmpAddition = [i for i in testList if i not in fullList]

        dateRes.append(j)
        patientRes.append(ssFolder)
        fileRes.append(tmpFile)
        fileNumRes.append(tmpFileNum)
        fileNotHave.append(tmpNotHave)
        fileAddition.append(tmpAddition)


#         print(tmpFile)


# In[18]:


import pandas as pd
resFileListPwd = './course/COMP5331//resFileListPwd_clean2.csv'


# In[19]:


resFileListPd = pd.DataFrame(list(zip(dateRes, patientRes, fileRes, fileNumRes, fileNotHave, fileAddition)),
                            columns=['date', 'patientNam', 'files', 'fileNum', 'fileNotHave', ' fileAddition'])


# In[20]:


resFileListPd[:10]


# In[21]:


resFileListPd.shape


# In[22]:


resFileListPd.to_csv(resFileListPwd)


# In[ ]:
