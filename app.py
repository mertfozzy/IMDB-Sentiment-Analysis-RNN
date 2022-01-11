import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from keras.datasets import imdb #dataseti import ediyoruz
from keras.preprocessing.sequence import pad_sequences # cümle sayılarını fixlemek için
from keras.models import Sequential # model yaratıp layerları dizi haline getireceğiz
from keras.layers.embeddings import Embedding # yoğunluk vektörlerine dönüşüm
from keras.layers import SimpleRNN, Dense, Activation # RNN, sınıflandırma ve sigmoid fonksiyonu

import warnings
warnings.filterwarnings("ignore")

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(path = "imdb.npz", #train ve test
                                                       num_words = None,
                                                       skip_top = 0, #en sık kullanılan kelimeleri ignore edelim mi?
                                                       maxlen = None, #yorumların kelime sayısını kırpsın mı?
                                                       seed = 113, #keras dökümanından geldi
                                                       start_char = 1, #yorumumuzdaki hangi karakterden başlasın? (dökümandan)
                                                       oov_char = 2, # default değeri 2
                                                       index_from = 3) #default değeri 3

"""=========================================================================="""

def testDataset():
    print("\n\n")
    print("\nX train shape: ", X_train.shape) 
    print("\nY train shape: ", Y_train.shape)
    print("\nY train values : ", np.unique(Y_train))
    print("\nY test values : ", np.unique(Y_test))

    unique, counts = np.unique(Y_train, return_counts = True)
    print("\nY train distribution: ", dict(zip(unique, counts)))
    unique, counts = np.unique(Y_test, return_counts = True)
    print("\nY test distribution: ", dict(zip(unique, counts)))
 
    print("\n\n")
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
    
"""=========================================================================="""

def kelimeSayisi():
    review_len_train = []
    review_len_test = []
    
    for rev1, rev2 in zip(X_train, X_test):
        review_len_train.append(len(rev1)) #X_train
        review_len_test.append(len(rev2)) #X_test
        
    sns.distplot(review_len_train, hist_kws = {"alpha" : 0.3})
    sns.distplot(review_len_test, hist_kws = {"alpha" : 0.3})
    
    print("\n\n")
    print("Train mean : ", np.mean(review_len_train)) 
    print("Train median : ", np.median(review_len_train))
    print("Train mode: ", stats.mode(review_len_train)) 
    
"""=========================================================================="""
def kacFarkliKelime():
    word_index = imdb.get_word_index()
    print("\n\n")
    print("\n Toplamda ", len(word_index), " farklı kelime var.")

"""=========================================================================="""

def hangiKelime():

    word_index = imdb.get_word_index()
    num = int(input("Kaç numaralı kelimenin karşılığını bulmak istersiniz : "))
    for keys, values in word_index.items(): 
        if values == num : #integer burada veriliyor
            print("Girdiğiniz", values, "sayısı şu kelimeye denk geliyor: ", keys)
            break

"""=========================================================================="""

def whatItSay(index): 
   word_index = imdb.get_word_index()
   reverse_index = dict([(value,key) for (key, value) in word_index.items()])
   decode_review = " ".join([reverse_index.get(i - 3, "!") for i in X_train[index]])
   print("\n\nGirdiğiniz index numarasına denk gelen yorum : \n\n", decode_review)
   
   if Y_train[index] == 0 :
       print ("\n\nOlumsuz bir yorum. : " , Y_train[index])

   elif Y_train[index] == 1 :
       print ("\n\nOlumlu bir yorum. : " , Y_train[index])


"""=========================================================================="""

def main():
    print("\n\nWelcome to the IMDB Sentiment Analysis!\n\n")
    
    print("1 == Veri setini test et. (Dengeli mi?)\n")
    print("2 == Veri setindeki kelime sayısının histogramını çiz.\n")  
    print("3 == Kaç farklı kelime var? \n")
    print("4 == Hangi sayı hangi kelimeye denk geliyor? \n")
    print("5 == Yorumu metine dönüştür. (List-Comparasion) \n")
    
    
    number1 = input("What do you want to monitor? : \t")
    
    if number1 == '1': 
        testDataset()
    
    elif number1 == '2': 
        kelimeSayisi()
 
    elif number1 == '3':
        kacFarkliKelime()
        
    elif number1 == '4':
        hangiKelime()
    
    elif number1 == '5':
        index = int(input("\nHerhangi bir index numarası giriniz : "))
        whatItSay(index)
        

main()

