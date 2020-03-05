from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
import json
import tensorflow
import numpy
import random
import keyboard

from tensorflow.keras.models import load_model

from tensorflow.python.keras import models
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D

TOP_K=2000
MAX_SEQUENCE_LENGTH= 500
tokenizer = text.Tokenizer(num_words=TOP_K, lower=True, split=' ',char_level=False)
def sequence_vector (train):
    g=[]
    for s in train:
        g.extend(s)
    print(g)
    tokenizer.fit_on_texts(train)
    x_train = tokenizer.texts_to_sequences(g)
    
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    return x_train, tokenizer.word_index

def label_vector(data):
    output = []
    tag_labels=[]
    num_tags=8


    out_empty = [0 for _ in range(8)]
    for i,item in enumerate(data['intents']):
        tag_labels.append(item['tag'])
        
        for patterm in item['patterns']:
            output_row = out_empty[:]
            output_row[i] = 1
            output.append(i)
    k= numpy.array(output)
    J = numpy.array([0,1,2,3,4,5,6,7])
    return k, tag_labels
def last_layer_units(num_tags):
    if num_tags == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'sigmoid'
        units = num_tags
    return units, activation

#sepCNN model
def sepcnn_model(blocks,
                 filters,
                 kernel_size,
                 embedding_dim,
                 dropout_rate,
                 pool_size,
                 input_shape,
                 num_tags,
                 num_features,
                 use_pretrained_embedding=False,
                 is_embedding_trainable=True,
                 embedding_matrix=None):

    mod_units, mod_activation = last_layer_units(num_tags)
    model = models.Sequential()

    model.add(Embedding(input_dim=num_features,
                        output_dim=embedding_dim,
                        input_length=input_shape[0], ))
        
    for _ in range(blocks-1):
        model.add(Dropout(rate=dropout_rate))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(MaxPooling1D(pool_size=pool_size))

    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(mod_units, activation=mod_activation))
    return model

def train_sequence_model(data, 
                        learning_rate = 1e-2,
                        epochs=1000,
                        batch_size=10,
                        blocks=3,
                        filter =32,
                        dropout_rate=0.4,
                        embedding_dim=200,
                        kernel_size=3,
                        pool_size=1):
    train =[]
    for intent in data["intents"]:
        
        train.append(intent['patterns'])
    print(train)
    labels, tag_labels= label_vector(data)


    x_train, word_index = sequence_vector(train)
    print(word_index)
    num_features = (len(word_index)+1)
    model = sepcnn_model(
        blocks=blocks,
        filters=filter,
        kernel_size=kernel_size,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        pool_size=pool_size,
        input_shape= x_train.shape[1:],
        num_features=num_features,
        num_tags=8,
    )


    loss = 'sparse_categorical_crossentropy'
    optimizer = tensorflow.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer='adam', loss=loss)
    
    history = model.fit(
            x_train,
            labels,
            epochs=epochs,
          
            batch_size=batch_size)
    model.summary()
    
    
    while True:
        print("Start talking with the bot (type quit to stop)!")
        inp = input("You: ")
        if keyboard.is_pressed('Esc') or inp =='quit':
            break
        
        x_val = tokenizer.texts_to_sequences([inp])

        x_val= sequence.pad_sequences(x_val, maxlen=1)
        result=model.predict(numpy.array(x_val))

        print(result)
        results_index = numpy.argmax(result[0])

        tag = tag_labels[results_index]
        
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
          
        print(random.choice(responses))


if __name__ == '__main__':

    with open("intents.json") as file:
        data = json.load(file)

    train_sequence_model(data)
        
