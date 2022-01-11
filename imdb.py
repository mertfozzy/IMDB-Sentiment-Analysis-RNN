"""
==============================================================================
IMDB Sentiment Analysis Project

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

#içeride ne kadar veri var? (25K)
print("X train shape: ", X_train.shape) 
print("Y train shape: ", Y_train.shape)




"""=====================Exploratory Data Science = EDA======================"""

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
insideX = X_train[0] #sıfırıncı indexi değere koyduk, içerisinde integer değerler var.
print(insideX)
print("Size of X_train[0]: ", len(insideX)) #toplamda kaç integer var?


#X_train ve X_test içerisinde dolaş, uzunlukları listeye koy. Bu bize tüm datasetteki yorumların kelime sayılarını gösterecek.
review_len_train = []
review_len_test = []

for rev1, rev2 in zip(X_train, X_test):
    review_len_train.append(len(rev1)) #X_train
    review_len_test.append(len(rev2)) #X_test
    
    
# Kelime sayılarının dağılımına bakalım, histogram çiziyoruz :
sns.distplot(review_len_train, hist_kws = {"alpha" : 0.3})
sns.distplot(review_len_test, hist_kws = {"alpha" : 0.3})

print("Train mean : ", np.mean(review_len_train)) #histogramın orta noktası
print("Train median : ", np.median(review_len_train)) #histogramın medyanı
print("Train mode: ", stats.mode(review_len_train)) #histogramın en tepe noktası


# hangi kelimeden kaç tane var : (burada kelimeler sayılara denk geliyor)
word_index = imdb.get_word_index()
print(type(word_index))
print(len(word_index))


#verilen sayının hangi kelimeye denk geldiğini bulalım : 
for keys, values in word_index.items(): 
    if values == 5 : #integer burada veriliyor
        print("The integer", values, "corresponds for the word: ", keys)


#yorumu metine dönüştürme (list-comparasion method)
# metod yazdım bu sayede yorumu parse edip olumlu olumsuz olarak da ayırt ediyoruz.
def whatItSay(index): #default olarak 24 verdim
   reverse_index = dict([(value,key) for (key, value) in word_index.items()])
   decode_review = " ".join([reverse_index.get(i - 3, "!") for i in X_train[index]]) #decoding yapıyoruz
   print(decode_review)
   print(Y_train[index])
   return decode_review

decoded_review = whatItSay(39) # yorum indexi parantezden veriliyor

"""================================End of EDA================================"""




"""================================Preprocess================================"""
#preprocess ile veri seti train edilebilir hale geliyor :

num_words = 15000 #kelime sayısını 15K ile sınırla
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=num_words)

maxlen = 130
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

print(X_train[5])

for i in X_train[0:10]:
    print(len(i))
    
decoded_review = whatItSay(36)

"""=========================================================================="""




"""==========================Recurrent Neural Network========================"""

rnn = Sequential() # layerlar eklenecek
rnn.add(Embedding(num_words, 32, input_length = len(X_train[0]))) # embeddingi (integerları belirli boyutlarda yoğunluk vektörlerine çevirmeye yarar) ekle
rnn.add(SimpleRNN(16, input_shape = (num_words, maxlen), return_sequences = False, activation = "relu")) #15K kelimem olucak, maximum 130 kelime olacak
rnn.add(Dense(1)) # sınıflandırma yapmak için layer ekledik
rnn.add(Activation("sigmoid")) # binary-classification yapmak için aktivasyon fonksiyonu ekledik

print(rnn.summary())
rnn.compile(loss = "binary_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])

# TRAIN : 
history = rnn.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 5, batch_size = 128, verbose = 1)

"""=========================================================================="""




score = rnn.evaluate(X_test, Y_test)
print("Accuracy : ", score[1]*100)

plt.figure()
plt.plot(history.history["accuracy"], label = "Train")
plt.plot(history.history["val_accuracy"], label = "Test")
plt.title("Acc")
plt.xlabel("Acc")
plt.ylabel("Epochs")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label = "Train")
plt.plot(history.history["val_loss"], label = "Test")
plt.title("Acc")
plt.xlabel("Acc")
plt.ylabel("Epochs")
plt.legend()
plt.show()





