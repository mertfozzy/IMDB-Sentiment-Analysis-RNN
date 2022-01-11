"""
==============================================================================
IMDB Sentiment Analysis Project

Student Name: Mert Altuntas | Student ID:1804010005 
==============================================================================

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from keras.datasets import imdb 
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential 
from keras.layers.embeddings import Embedding 
from keras.layers import SimpleRNN, Dense, Activation 

warnings.filterwarnings("ignore")

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(path = "imdb.npz", 
                                                       num_words = None,
                                                       skip_top = 0, 
                                                       maxlen = None, 
                                                       seed = 113, 
                                                       start_char = 1, 
                                                       oov_char = 2, 
                                                       index_from = 3) 

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
def machineLearning():
    #PREPROCESS:
    print("\n\nPreprocess işlemleri başlatılıyor :")
    num_words = 15000 
    (X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=num_words)
    maxlen = 130
    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)
    print("\n\nPreprocess başarı ile tamamlandı..")
    
    #BUILDING RECURRENT NEURAL NETWORK :
    print("\n\nRecurrent Neural Network kuruluyor :")
    rnn = Sequential() 
    rnn.add(Embedding(num_words, 32, input_length = len(X_train[0]))) 
    rnn.add(SimpleRNN(16, input_shape = (num_words, maxlen), return_sequences = False, activation = "relu")) 
    rnn.add(Dense(1)) 
    rnn.add(Activation("sigmoid")) 
    print("\n\n")
    print(rnn.summary())
    rnn.compile(loss = "binary_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])
    print("\n\nRecurrent Neural Network başarı ile kuruldu..")
    
    #TRAINING RECURRENT NEURAL NETWORK :
    print("\n\nOluşturulan model eğitiliyor :")
    history = rnn.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 5, batch_size = 128, verbose = 1)
    score = rnn.evaluate(X_test, Y_test)
    print("\nEğitim tamamlandı... \nAccuracy : ", score[1]*100)
    
    #SONUÇLAR : 
    print("\n\nSonuçlar şu şekilde : \n\n")
    plt.figure()
    plt.plot(history.history["accuracy"], label = "Train")
    plt.plot(history.history["val_accuracy"], label = "Test")
    plt.title("Accuracy")
    plt.xlabel("Accuracy")
    plt.ylabel("Epochs")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(history.history["loss"], label = "Train")
    plt.plot(history.history["val_loss"], label = "Test")
    plt.title("Loss")
    plt.xlabel("Loss")
    plt.ylabel("Epochs")
    plt.legend()
    plt.show()

"""=========================================================================="""

def main():
    print("\n\nWelcome to the IMDB Sentiment Analysis!\n")
    print("\n\nExploratory Data Science :\n")
    print("\t1 ==> Veri setini test et. (Dengeli mi?)\n")
    print("\t2 ==> Veri setindeki kelime sayısının histogramını çiz.\n")  
    print("\t3 ==> Kaç farklı kelime var? \n")
    print("\t4 ==> Hangi sayı hangi kelimeye denk geliyor? \n")
    print("\t5 ==> Yorumu metine dönüştür. (List-Comparasion) \n")
    print("\nMakine Öğrenmesi Aşamaları ve RNN :\n")
    print("\t6 ==> Modeli eğitmeye başlayın. (Otomatik) \n")
    
    
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
    
    elif number1 == '6':
        machineLearning()
        
main()

