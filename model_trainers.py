from keras.models import Sequential
from keras import layers
from keras.models import load_model
import numpy as np
from six.moves import range

import matplotlib.pyplot as plt


def train_add_model(train_x, train_y, val_x, val_y, digits, hidden_size, batch_size, layer, epoch, chars, save_model=False):
    
    print('Build model...')

    model = Sequential()
    model.add(layers.LSTM(hidden_size, input_shape=(2*digits + 1, len(chars))))
    model.add(layers.RepeatVector(digits + 1))
    for _ in range(layer):
        model.add(layers.LSTM(hidden_size, return_sequences=True))

    model.add(layers.TimeDistributed(layers.Dense(len(chars))))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(train_x, train_y,
                  batch_size=batch_size,
                  epochs=epoch,
                  validation_data=(val_x, val_y),
                  verbose=False
                  )
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['training_acc', 'val_acc'])
    plt.ylim(0, 1.1)
    plt.show()
    
    train_size = len(train_x)
    model_name = str(digits)+'_digits_add_'+str(epoch)+'_epochs_'+str(train_size)+'_train_'+str(layer)+'_layers'
    
    if save_model:
        model.save('./models/'+model_name+'.h5')
        
    return model

def train_sub_model(train_x, train_y, val_x, val_y, digits, hidden_size, batch_size, layer, epoch, chars, save_model=False):
    
    print('Build model...')

    model = Sequential()
    model.add(layers.LSTM(hidden_size, input_shape=(2*digits + 1, len(chars))))
    model.add(layers.RepeatVector(digits))
    for _ in range(layer):
        model.add(layers.LSTM(hidden_size, return_sequences=True))

    model.add(layers.TimeDistributed(layers.Dense(len(chars))))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(train_x, train_y,
                  batch_size=batch_size,
                  epochs=epoch,
                  validation_data=(val_x, val_y),
                  verbose=False
                  )
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['training_acc', 'val_acc'])
    plt.ylim(0, 1.1)
    plt.show()
    
    train_size = len(train_x)
    model_name = str(digits)+'_digits_sub_'+str(epoch)+'_epochs_'+str(train_size)+'_train_'+str(layer)+'_layers'
    
    if save_model:
        model.save('./models/'+model_name+'.h5')
        
    return model

def train_comb_model(train_x, train_y, val_x, val_y, digits, hidden_size, batch_size, layer, epoch, chars, save_model=False):
    
    print('Build model...')

    model = Sequential()
    model.add(layers.LSTM(hidden_size, input_shape=(2*digits + 1, len(chars))))
    model.add(layers.RepeatVector(digits + 1))
    for _ in range(layer):
        model.add(layers.LSTM(hidden_size, return_sequences=True))

    model.add(layers.TimeDistributed(layers.Dense(len(chars))))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(train_x, train_y,
                  batch_size=batch_size,
                  epochs=epoch,
                  validation_data=(val_x, val_y),
                  verbose=False
                  )
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['training_acc', 'val_acc'])
    plt.ylim(0, 1.1)
    plt.show()
    
    train_size = len(train_x)
    model_name = str(digits)+'_digits_comb_'+str(epoch)+'_epochs_'+str(train_size)+'_train_'+str(layer)+'_layers'
    
    if save_model:
        model.save('./models/'+model_name+'.h5')
        
    return model