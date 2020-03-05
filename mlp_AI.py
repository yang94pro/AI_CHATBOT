import json
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense, Dropout, Flatten


# Vectorization parameters
# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 0
def get_data():
    with open("intents.json") as file:
        data = json.load(file)

    train_words= []
    train_data = []
    train_labels = []
    train_y =[]
    for i, item in enumerate(data['intents']):
        for pattern in item['patterns']:
            train_words.append(pattern)
            train_y.append(i)

        train_data.append(item['patterns'])
        train_labels.append(item['tag'])
    print(train_data)
    print(train_y)

    return (train_data, train_y)

def ngram_vectorize(train_data, train_labels,):
    """Vectorizes texts as ngram vectors.
    1 text = 1 tf-idf vector the length of vocabulary of uni-grams + bi-grams.
    # Arguments
        train_data: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.
    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            
    }
    vectorizer = TfidfVectorizer(**kwargs)
    x_train=[]
    # Learn vocabulary from training texts and vectorize training texts.
    for data in train_data:
        x_train.append(vectorizer.fit_transform(data))

  
    x_train = numpy.array(x_train)
    

    return x_train


def mp_model (layers, units, dropout_rate, input_shape, num_tags):
    op_units = 8
    op_activation = 'softmax'
    model = models.Sequential()
   
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))
    model.add(Flatten())
    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation= op_activation))
    return model

def train_model(data,
                learning_rate=1e-3,
                epochs=100,
                batch_size=100,
                layers=2,
                units=31,
                dropout_rate=0.2):
    
    (train_data, train_y) = data
    num_tags = 8
    train_y = numpy.array(train_y)
    x_train=ngram_vectorize(train_data, train_y)
    x_train = numpy.array(x_train)

    model=mp_model(layers=layers,
                    units=units,
                    dropout_rate=dropout_rate,
                    input_shape=x_train.shape[1:],
                    num_tags=num_tags)
    loss = 'sparse_categorical_crossentropy'
    model.compile( optimizer= Adam(lr=learning_rate), loss=loss)
    Y = numpy.array([0,1,2,3,4,5,6,7])

    model.fit(
        x_train,
        Y,
        epochs=epochs,
        verbose=2,
        batch_size=batch_size
    )
    model.summary()



if __name__ == '__main__':

    data =get_data()
    train_model(data)
