import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import multiprocessing as mp
import load_npy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import vae_model
tf.config.threading.set_intra_op_parallelism_threads(800)

'''
# Reference
- https://arxiv.org/pdf/1606.05908.pdf
- https://keras.io/examples/generative/vae/
- https://sites.google.com/illinois.edu/supervised-vae
- https://www.linkedin.com/pulse/supervised-variational-autoencoder-code-included-ibrahim-sobh-phd
- https://medium.com/analytics-vidhya/activity-detection-using-the-variational-autoencoder-d2b017da1a88
'''

if __name__ == '__main__':
    # train the model
    latent_dim = 2
    encoder, decoder, clf = defineModel(latent_dim)
    x_train = x_train.reshape(x_train.shape[0], 48, 52, 1)
    x_test = x_test.reshape(x_test.shape[0],48,52,1)
    pd.DataFrame(y1_train)[0].value_counts()
    vae = SVAE(encoder, decoder, clf)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(x_train,tf.keras.utils.to_categorical(pd.DataFrame(y1_train)[0].astype('category').cat.codes,num_classes=18),epochs=30, batch_size=128)
    z_mean, _, _ = vae.encoder.predict(x_test)
    mappingLabel = pd.DataFrame({'label':pd.DataFrame(y1_train)[0],
                                'category':pd.DataFrame(y1_train)[0].astype('category').cat.codes})
    uniMappingLabel = mappingLabel.drop_duplicates() # to change into healthy and unhealthy
    uniMappingLabel['healthyLabel'] = ['unhealthy' for i in range(9)] + ['healthy' for i in range(9)]
    pret_label = vae.clf(z_mean)

    # get the predict label
    predictLabel = myGetLabel(pret_label)
    trueLabel = y1_test
    for i in uniMappingLabel['label'].index:
        trueLabel[trueLabel == uniMappingLabel['label'][i]] = uniMappingLabel['category'][i]
    trueLabel = trueLabel.astype('int')
    y_pred = np.array(predictLabel)
    y_true = np.array(trueLabel)
    labelDict = dict(zip(uniMappingLabel['label'].values,uniMappingLabel['category'].values))
    y_pred_binary = np.array(y_pred)
    y_true_binary = np.array(y_true)
    healthyRange = uniMappingLabel.loc[uniMappingLabel['healthyLabel']=='healthy','category'].values
    unhealthyRange = uniMappingLabel.loc[uniMappingLabel['healthyLabel']=='unhealthy','category'].values
    for i in tqdm(range(len(y_pred_binary))):
        if y_pred_binary[i] in healthyRange:
            y_pred_binary[i] = 1
        elif y_pred_binary[i] in unhealthyRange:
            y_pred_binary[i] = 0
        if y_true_binary[i] in healthyRange:
            y_true_binary[i] = 1
        elif y_true_binary[i] in unhealthyRange:
            y_true_binary[i] = 0

    # calculate the metrics and print
    print([precision_score(y_true, y_pred, average='micro'),
    recall_score(y_true, y_pred, average='micro'),
    accuracy_score(y_true, y_pred),
    f1_score(y_true, y_pred, average='micro')])
    print([precision_score(y_true_binary, y_pred_binary, average='micro'),
    recall_score(y_true_binary, y_pred_binary, average='micro'),
    accuracy_score(y_true_binary, y_pred_binary),
    f1_score(y_true_binary, y_pred_binary, average='micro')])

    # print the latent space
    plot_label_clusters(vae, x_test, pd.DataFrame(y1_test)[0].astype('category').cat.codes)
    plot_label_clusters(vae, x_test, y_true_binary)
