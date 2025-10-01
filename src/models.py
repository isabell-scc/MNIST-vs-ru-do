import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from src.data import data_prep

#seed
SEED = 42
keras.utils.set_random_seed(SEED)

X_train_mlp, X_test_mlp, X_train_cnn, X_test_cnn, y_train, y_test = data_prep(nivel_ruido=0.0)

#Conversão das labels
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

def mlp_model():
    model_mlp = Sequential([
        Dense(512, activation='relu', input_shape=(784,)),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])

    model_mlp.compile(loss='categorical_crossentropy',
                      optimizer= 'adam',
                      metrics=['accuracy'])

    return model_mlp


def cnn_model(input_shape= (28, 28, 1), num_classes = 10 ):
    model_cnn = Sequential([
        #bloco1
        Conv2D(32, (3, 3), activation='relu', input_shape= input_shape, padding='same'),
        MaxPooling2D((2, 2)),
        #bloco2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.40),
        Dense(num_classes, activation='softmax'),

    ])

    model_cnn.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])


    return model_cnn
    
def evaluate_model(model, X_test, y_test):
    nivel = [0.0, 0.1, 0.3, 0.5]
    
    resultados = {}
    for nivel in nivel:

        X_train_mlp, X_test_mlp, X_train_cnn, X_test_cnn, y_train, y_test = data_prep(nivel_ruido=nivel)

        #Conversão das labels
        y_train = keras.utils.to_categorical(y_train, num_classes=10)
        y_test = keras.utils.to_categorical(y_test, num_classes=10)

        if model == mlp_model:
            loss, acc = model.evaluate(X_test_mlp, y_test, verbose=0)
            resultados[nivel] = {'loss': loss, 'accuracy': acc}
            print(f"Ruído {nivel}: Acurácia = {acc:.4f}, Perda = {loss:.4f}")


        if model == cnn_model:
            loss, acc = model.evaluate(X_test_cnn, y_test, verbose=0)
            resultados[nivel] = {'loss': loss, 'accuracy': acc}
            print(f"Ruído {nivel}: Acurácia = {acc:.4f}, Perda = {loss:.4f}")
            
    return resultados

