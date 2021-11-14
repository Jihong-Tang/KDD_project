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
tf.config.threading.set_intra_op_parallelism_threads(800)

'''
# Reference
- https://arxiv.org/pdf/1606.05908.pdf
- https://keras.io/examples/generative/vae/
- https://sites.google.com/illinois.edu/supervised-vae
- https://www.linkedin.com/pulse/supervised-variational-autoencoder-code-included-ibrahim-sobh-phd
- https://medium.com/analytics-vidhya/activity-detection-using-the-variational-autoencoder-d2b017da1a88
'''

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class SVAE(keras.Model):
    def __init__(self, encoder, decoder, clf, **kwargs):
        super(SVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.clf = clf
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.clf_loss_tracker = keras.metrics.Mean(name='clf_loss')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            classify_z = self.clf(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(x, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            clf_loss = tf.keras.losses.categorical_crossentropy(y, classify_z)
            total_loss = clf_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.clf_loss_tracker.update_state(clf_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "clf_loss": self.clf_loss_tracker.result(),
        }

def defineModel(latent_dim):
    encoder_inputs = keras.Input(shape=(48, 52, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(12 * 13 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((12, 13, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    clf_latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling_clf')
    clf_outputs = layers.Dense(18, activation='softmax', name='class_output')(clf_latent_inputs)
    clf = keras.Model(clf_latent_inputs, clf_outputs, name='clf')
    return(encoder, decoder, clf)

def myGetLabel(narr):
    resLabel = []
    for i in tqdm(range(narr.shape[0])):
        resLabel.append([i for i, x in enumerate(pret_label[i] == max(pret_label[i])) if x][0])
    return(resLabel)

def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
