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

warnings.filterwarnings("ignore") # just ignores unnecessary warnings

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

    print("\nX train shape: ", X_train.shape) 
    print("\nY train shape: ", Y_train.shape)
    print("\nY train values : ", np.unique(Y_train))
    print("\nY test values : ", np.unique(Y_test))

    unique, counts = np.unique(Y_train, return_counts = True)
    print("\nY train distribution: ", dict(zip(unique, counts)))
    unique, counts = np.unique(Y_test, return_counts = True)
    print("\nY test distribution: ", dict(zip(unique, counts)))
 
    print("\n")
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
    
    plt.pause(0.05)
    

    
"""=========================================================================="""

def kelimeSayisi():
    review_len_train = []
    review_len_test = []
    
    for rev1, rev2 in zip(X_train, X_test):
        review_len_train.append(len(rev1)) #X_train
        review_len_test.append(len(rev2)) #X_test
        
    sns.distplot(review_len_train, hist_kws = {"alpha" : 0.3})
    sns.distplot(review_len_test, hist_kws = {"alpha" : 0.3})
    
    plt.pause(0.05)
    
    print("\n")
    print("Train mean : ", np.mean(review_len_train)) 
    print("Train median : ", np.median(review_len_train))
    print("Train mode: ", stats.mode(review_len_train)) 
    
"""=========================================================================="""
def kacFarkliKelime():
    word_index = imdb.get_word_index()
    print("\n\tToplamda ", len(word_index), " farkl?? kelime var.")

"""=========================================================================="""

def hangiKelime():

    word_index = imdb.get_word_index()
    num = int(input("Ka?? numaral?? kelimenin kar????l??????n?? bulmak istersiniz : "))
    for keys, values in word_index.items(): 
        if values == num : #integer burada veriliyor
            print("Girdi??iniz", values, "say??s?? ??u kelimeye denk geliyor: ", keys)
            break

"""=========================================================================="""

def whatItSay(index): 
   word_index = imdb.get_word_index()
   reverse_index = dict([(value,key) for (key, value) in word_index.items()])
   decode_review = " ".join([reverse_index.get(i - 3, "!") for i in X_train[index]])
   print("\nGirdi??iniz index numaras??na denk gelen yorum : \n\n", decode_review)
   
   if Y_train[index] == 0 :
       print ("\n\nOlumsuz bir yorum. : " , Y_train[index])

   elif Y_train[index] == 1 :
       print ("\n\nOlumlu bir yorum. : " , Y_train[index])


"""=========================================================================="""
def machineLearning():
    #PREPROCESS:
    print("\n\n==>Preprocess i??lemleri ba??lat??l??yor :")
    num_words = 15000 
    (X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=num_words)
    maxlen = 130
    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)
    print("\n\nPreprocess ba??ar?? ile tamamland??..")
    
    #BUILDING RECURRENT NEURAL NETWORK :
    print("\n\n==>Recurrent Neural Network kuruluyor :")
    rnn = Sequential() 
    rnn.add(Embedding(num_words, 32, input_length = len(X_train[0]))) 
    rnn.add(SimpleRNN(16, input_shape = (num_words, maxlen), return_sequences = False, activation = "relu")) 
    rnn.add(Dense(1)) 
    rnn.add(Activation("sigmoid")) 
    print("\n\n")
    print(rnn.summary())
    rnn.compile(loss = "binary_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])
    print("\n\nRecurrent Neural Network ba??ar?? ile kuruldu..")
    
    #TRAINING RECURRENT NEURAL NETWORK :
    print("\n\n==>Olu??turulan model e??itiliyor :")
    history = rnn.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 5, batch_size = 128, verbose = 1)
    score = rnn.evaluate(X_test, Y_test)
    print("\nE??itim tamamland??... \nAccuracy : ", score[1]*100)
    
    #SONU??LAR : 
    print("\n\n==>Sonu??lar ??u ??ekilde : \n")
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
    
    plt.pause(0.05)

"""=========================================================================="""

def main():
    
    print("\n\nWelcome to the IMDB Sentiment Analysis!\n")
    
    while(1) :
        
        print("\n\nExploratory Data Science :\n")
        print("\t1 ==> Veri setini test et. (Dengeli mi?)\n")
        print("\t2 ==> Veri setindeki kelime say??s??n??n histogram??n?? ??iz.\n")  
        print("\t3 ==> Ka?? farkl?? kelime var? \n")
        print("\t4 ==> Hangi say?? hangi kelimeye denk geliyor? \n")
        print("\t5 ==> Yorumu metine d??n????t??r. (List-Comparasion) \n")
        print("\nMakine ????renmesi A??amalar?? ve RNN :\n")
        print("\t6 ==> Modeli e??itmeye ba??lay??n. (Otomatik) \n")
        
        number = input("\n----> What do you want to monitor? : \t")
        
        if number == '1': 
            testDataset()
        
        elif number == '2': 
            kelimeSayisi()
     
        elif number == '3':
            kacFarkliKelime()
            
        elif number == '4':
            hangiKelime()
        
        elif number == '5':
            index = int(input("\nHerhangi bir index numaras?? giriniz : "))
            whatItSay(index)
        
        elif number == '6':
            machineLearning()
        
main()

