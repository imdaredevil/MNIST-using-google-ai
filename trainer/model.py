import tensorflow
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D, BatchNormalization, Dropout, Flatten
from tensorflow.keras import Model, Sequential

def create_model(optimizer, loss):
    my_model = Sequential()
    my_model.add(Conv2D(32,5,activation='relu',padding='same',input_shape=(28,28,1)))
    my_model.add(BatchNormalization())
    my_model.add(MaxPool2D())
    my_model.add(Dropout(0.4))
    my_model.add(Conv2D(32,5,activation='relu', padding='same'))
    my_model.add(BatchNormalization())
    my_model.add(MaxPool2D())
    my_model.add(Dropout(0.4))
    my_model.add(Conv2D(64,3, activation='relu',padding='same'))
    my_model.add(BatchNormalization())
    my_model.add(MaxPool2D(padding='same'))
    my_model.add(Dropout(0.4))
    my_model.add(Flatten())
    my_model.add(Dense(64, activation='relu'))
    my_model.add(BatchNormalization())
    my_model.add(Dropout(0.4))
    my_model.add(Dense(128, activation='relu'))
    my_model.add(BatchNormalization())
    my_model.add(Dropout(0.4))
    my_model.add(Dense(10))
    print(my_model.summary())
    
    my_model.compile(optimizer=optimizer, loss=loss)
    return my_model
