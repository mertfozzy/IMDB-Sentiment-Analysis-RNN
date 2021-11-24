"""
==============================================================================
Student Name: Mert Altuntas | Student ID:1804010005 
==============================================================================

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.datasets import imdb #dataseti import ediyoruz
from keras.preprocessing.sequence import pad_sequences # cümle sayılarını fixlemek için
from keras.models import Sequential # model yaratıp layerları dizi haline getireceğiz
from keras.layers.embeddings import Embedding # yoğunluk vektörlerine dönüşüm
from keras.layers import SimpleRNN, Dense, Activation # RNN, sınıflandırma ve sigmoid fonksiyonu

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(path = "imdb.npz",
                                                       num_words = None,
                                                       skip_top = 0,
                                                       maxlen = None,
                                                       seed = 113,
                                                       start_char = 1,
                                                       oov_char = 2,
                                                       index_from = 3)