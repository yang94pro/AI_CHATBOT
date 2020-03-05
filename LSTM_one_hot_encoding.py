import json
import pickle
# import tflearn
# import tensorflow
import random

import keyboard
import nltk
import numpy
import tensorflowjs as tfjs
from nltk.stem.snowball import SnowballStemmer
# from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model

# vectorizer = CountVectorizer()
stemmer = SnowballStemmer("english")




with open("intents.json") as file:
    data = json.load(file)

try:
    x
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    print(words)
    print(docs_x)
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

    training = numpy.array(training)
    print(len(training))
    output = numpy.array(output)
    data_dim = len(words)
    timesteps=1
    model = Sequential()
    x_train = numpy.reshape(training, (training.shape[0],1,training.shape[1]))
    output = numpy.reshape(output, (output.shape[0],1,output.shape[1]))
    print(x_train)
    model.add(LSTM(len(words), input_shape=(timesteps, data_dim), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(LSTM(len(words), input_shape=(timesteps, data_dim), return_sequences=True))
    model.add(Dense(25))

    model.add(Dense(len(labels), activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.summary()
    model.fit(x_train, output, epochs=300, batch_size=100)
    model.save("model.h5")
    tfjs.converters.save_keras_model(model, '/')
    print("Saved model to disk")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
   
    

    return numpy.array(bag)

def chat():
    print("Start talking with the bot (type quit to stop)!")
    try:
        x
        model = load_model('model.h5')
    except:
        pass
    while True:
        inp = input("You: ")
        if keyboard.is_pressed('Esc') or inp == 'quit':
            break
        
        
        inpw =numpy.asarray([bag_of_words(inp, words)])
        inpw= numpy.reshape(inpw, (inpw.shape[0],1,inpw.shape[1]))
        results = model.predict(inpw)
        results_index = numpy.argmax(results)
        print(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()
