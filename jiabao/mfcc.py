#!/usr/bin/env python
# coding: utf-8

# In[1]:


# remain line
# this notebook is for the


# In[1]:


# install the package
# !pip install python_speech_features


# In[2]:


# This is the notebook for calculating the MFCC of the wmv
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import os
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# make the figure clear
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 300

from matplotlib import rcParams
rcParams['font.family']


# In[4]:


rcParams['font.sans-serif']


# In[5]:


# set the font family
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Helvetica'


# In[ ]:





# In[ ]:





# # Read the files and calculate the MFCC

# In[8]:



audioPwd = './course/COMP5331/testAudio/vowel-a.wav'
(rate,sig) = wav.read(audioPwd)
mfcc_feat = mfcc(sig,rate, nfft=2048) # increase the nfft
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(sig,rate, nfft=2048)

print(fbank_feat[1:3,:])


# In[9]:


print(fbank_feat[:,:])


# In[10]:


mfcc_feat


# In[14]:


plt.plot(sig)


# In[15]:


imgplot = plt.imshow(mfcc_feat.T, aspect='auto')


# In[17]:


mfcc_feat.shape


# In[19]:


d_mfcc_feat.shape


# In[30]:


fbank_feat.shape


# In[22]:


imgplot2 = plt.imshow(d_mfcc_feat.T, aspect='auto')


# In[33]:


imgplot3 = plt.imshow(fbank_feat.T, aspect='auto')


# In[25]:


np.var(mfcc_feat)


# In[26]:


np.var(d_mfcc_feat)


# In[34]:


np.var(fbank_feat)


# In[27]:


np.var(mfcc_feat[1,:])


# In[29]:


np.var(d_mfcc_feat[1,:])


# In[35]:


np.var(fbank_feat[1,:])


# # Small loop: write the mfcc array to dat and small loop for all the audios

# In[46]:


audioPwd = './course/COMP5331/testAudio/'
fileList = os.listdir(audioPwd)
audioList = [i for i in fileList if 'wav' in i]

for i in audioList:
    audioPwd = os.path.join(audioPwd, i)

    resLcmfMfcc = 'lcmfMfcc.'+ i[:-4] + '.dat'

    print([audioPwd, resLcmfMfcc])



# In[49]:


audioPwd = './course/COMP5331/testAudio/'
fileList = os.listdir(audioPwd)
audioList = [i for i in fileList if 'wav' in i]

for i in audioList:
    audioPwdForPatient = os.path.join(audioPwd, i)

    resLcmfMfcc = 'lcmfMfcc.'+ i[:-4] + '.dat'
    resLcmfMfccPwd = os.path.join(audioPwd, resLcmfMfcc)

    (rate,sig) = wav.read(audioPwdForPatient)
    # mfcc_feat = mfcc(sig,rate, nfft=2048) # increase the nfft
    # d_mfcc_feat = delta(mfcc_feat, 2)
    fbank_feat = logfbank(sig,rate, nfft=2048)

    print(fbank_feat.shape)

    # write the csv
#     pd.DataFrame(fbank_feat).to_csv(resLcmfMfccPwd,
#                                     header=False,
#                                     index=False)



# In[53]:





# In[ ]:





# # Large loop: calculate & write MFCC.dats for all the patients

# In[10]:


from tqdm import tqdm
for i in tqdm(range(10000000)):
    pass


# In[11]:


allFolderList = [i for i in os.listdir('./course/COMP5331/Coswara-Data/') if '20' in i]

maxShape = 0

for j in allFolderList:
    pwd = os.path.join('./course/COMP5331/Coswara-Data/',j,j)

    subfolderList = os.listdir(pwd)

    print(pwd)
    for ssFolder in subfolderList:
        if 'csv' in ssFolder or 'DS_Store' in ssFolder:
            continue

        ssPwd = os.path.join(pwd, ssFolder) # patientFolder
        tmpFile = os.listdir(ssPwd) # files in the patientFolder
        audioPwd = ssPwd
        fileList = os.listdir(audioPwd)
        audioList = [i for i in fileList if 'wav' in i]

        for i in audioList:
            audioPwdForPatient = os.path.join(audioPwd, i)

            resLcmfMfcc = 'lcmfMfcc.'+ i[:-4] + '.dat'
            resLcmfMfccPwd = os.path.join(audioPwd, resLcmfMfcc)

            try:
                (rate,sig) = wav.read(audioPwdForPatient)
                # mfcc_feat = mfcc(sig,rate, nfft=2048) # increase the nfft
                # d_mfcc_feat = delta(mfcc_feat, 2)
                fbank_feat = logfbank(sig,rate, nfft=4800)

    #             print(fbank_feat.shape)
                maxShape = max(maxShape, fbank_feat.shape[0])

                # write the csv
                pd.DataFrame(fbank_feat).to_csv(resLcmfMfccPwd,
                                                header=False,
                                                index=False)

            except:
                print("Error in " + audioPwdForPatient)





print(maxShape)


# In[63]:


audioPwdForPatient


# In[58]:


ssPwd


# In[57]:


tmpFile


# In[ ]:





# In[ ]:





# In[ ]:
