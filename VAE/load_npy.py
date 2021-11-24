import numpy as np
import requests

def load_npy():
    # the dataset url
    url = 'https://drive.google.com/file/d/19wrDXxdr-3PQ0oSVNTDB3XPJqlmqsQTv/view?usp=sharing'
    r = requests.get(url, allow_redirects=True)
    open('dataAR.npy', 'wb').write(r.content)
    trainCut = int(np.load('dataAR.npy').shape[0]*0.8)
    dataAR = np.load('dataAR.npy')
    labelAR = np.load('labelAR.npy')
    hlabelAR = np.load('hlabelAR.npy')
    x_train = dataAR[:trainCut]
    x_test = dataAR[trainCut:]
    y1_train = labelAR[:trainCut]
    y1_test = labelAR[trainCut:]
    y2_train = hlabelAR[:trainCut]
    y2_test = hlabelAR[trainCut:]

load_npy()
