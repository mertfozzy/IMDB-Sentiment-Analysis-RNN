"""
==============================================================================
Student Name: Mert Altuntas | Student ID:1804010005 
==============================================================================

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from keras.datasets import imdb #dataseti import ediyoruz
from keras.preprocessing.sequence import pad_sequences # cümle sayılarını fixlemek için
from keras.models import Sequential # model yaratıp layerları dizi haline getireceğiz
from keras.layers.embeddings import Embedding # yoğunluk vektörlerine dönüşüm
from keras.layers import SimpleRNN, Dense, Activation # RNN, sınıflandırma ve sigmoid fonksiyonu

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(path = "imdb.npz", #train ve test
                                                       num_words = None,
                                                       skip_top = 0, #en sık kullanılan kelimeleri ignore edelim mi?
                                                       maxlen = None, #yorumların kelime sayısını kırpsın mı?
                                                       seed = 113, #keras dökümanından geldi
                                                       start_char = 1, #yorumumuzdaki hangi karakterden başlasın? (dökümandan)
                                                       oov_char = 2, # default değeri 2
                                                       index_from = 3) #default değeri 3

print("Type : ", type(X_train))
print("Type : ", type(Y_train))

print("X train shape: ", X_train.shape)
print("Y train shape: ", Y_train.shape)


# Exploratory Data Science = EDA

#unique metodu ile Y_train ve Y_test içerisinde neler var bakıyoruz :
print("Y train values : ", np.unique(Y_train))
print("Y test values : ", np.unique(Y_test))

# burada unique 0 negatif, 1 pozitif görüşler. eşit çıkıyorsa dengeli bir veri seti.
unique, counts = np.unique(Y_train, return_counts = True)
print("Y train distribution: ", dict(zip(unique, counts)))

unique, counts = np.unique(Y_test, return_counts = True)
print("Y test distribution: ", dict(zip(unique, counts)))

# bulduğumuz değerleri tabloya çeviriyoruz :
plt.figure()
sns.countplot(Y_train)
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.title("Y Train")

plt.figure()
sns.countplot(Y_test)
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.title("Y Test")

# X_train içinde neler var? :
a = X_train[0] #sıfırıncı indexi değere koyduk, içerisinde integer değerler var.
print(a)
print("Size of X_train[0]: ", len(a)) #toplamda kaç integer var?


#X_train ve X_test içerisinde dolaş, uzunlukları listeye koy. Bu bize tüm datasetteki yorumların kelime sayılarını gösterecek.
review_len_train = []
review_len_test = []

for i, ii in zip(X_train, X_test):
    review_len_train.append(len(i)) #X_train
    review_len_test.append(len(ii)) #X_test
    
    
# Kelime sayılarının dağılımına bakalım, histogram çiziyoruz :
sns.distplot(review_len_train, hist_kws = {"alpha" : 0.3})
sns.distplot(review_len_test, hist_kws = {"alpha" : 0.3})

print("Train mean : ", np.mean(review_len_train)) #histogramın orta noktası
print("Train median : ", np.median(review_len_train)) #histogramın medyanı
print("Train mode: ", stats.mode(review_len_train)) #histogramın en tepe noktası