from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import sklearn
import matplotlib

import os

import numpy as np

# For DataFrame object
import pandas as pd
import keras

# Neural Network
from keras.metrics import Precision, Recall
from keras.models import Sequential
from keras.layers import Dense, Dropout

# optimiser
optima = tf.keras.optimizers.RMSprop

# Text Vectorizing
from keras.preprocessing.text import Tokenizer

# Train-test-split
from sklearn.model_selection import train_test_split

# History visualization
from matplotlib import pyplot as plt

# Normalize
from sklearn.preprocessing import normalize

path = 'C:/train_final.csv'         #Путь до набора данных для обучения
df = pd.read_csv(path, encoding="utf-8")            #Перевод данных в датафрейм
df.head()           #Вывод верхней части датафрейма для проверки

def delete_new_line_symbols(text):          #Функция для удаления символов переноса строки
    text = text.replace('\n', ' ')
    return text

df['comment'] = df['comment'].apply(delete_new_line_symbols)        #Применяется функция
df.head()           #Вывод верхней части датафрейма для проверки
df.dropna()         #Удаление всех пустых значений при их наличии

df["toxic"] = df["toxic"].fillna(0)         #Заполнение пустых значений нулями

target = np.array(df['toxic'].astype('uint8'))          #Преобразование датафрейма в numpy array

tokenizer = Tokenizer(num_words=24000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',          #токенайзер
                      lower=True,
                      split=' ',
                      char_level=False)

tokenizer.fit_on_texts(df['comment'])           #Применение токенайзера

"""word_index = tokenizer.word_index            Необходимо для отладки (для определения словаря модели)
print(word_index)"""

matrix = tokenizer.texts_to_matrix(df['comment'], mode='count')         #Создание матрицы

switch = True

if not switch:
    def get_model():
        model = Sequential()

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=optima(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

        return model


    X = normalize(matrix)           #Нормализация матрицы
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)            #Деление датасета на выборки

    #print(X_train.shape, y_train.shape)

    model = get_model()         #присвоение переменной модели

    history = model.fit(X_train,               #Запуск модели с установленными параметрами
                        y_train,
                        epochs=12,
                        batch_size=1000,
                        validation_data=(X_test, y_test))

    history = history.history
    print(history.keys())

    fig = plt.figure(figsize=(20, 10))          #Инициализация модели

    ax1 = fig.add_subplot(221)              #Создание сабплотов для графики
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    x = range(12)

    ax1.plot(x, history["accuracy"], 'b-', label='Accuracy')            #График точности
    ax1.plot(x, history['val_accuracy'], 'r-', label='Validation accuracy')
    ax1.legend(loc='lower right')

    ax2.plot(x, history['val_loss'], 'r-', label='Losses')          #График потерь
    ax2.legend(loc='upper right')

    ax3.plot(x, history['val_precision'], 'b-', label='precision_class_1')          #График precision
    ax3.plot(x, history['precision'], 'r-', label='precision_class_2')
    ax3.legend(loc='upper left')

    ax4.plot(x, history['val_recall'], 'b-', label='recall_class_1')            #График recall
    ax4.plot(x, history['recall'], 'r-', label='recall_class_2')
    ax4.legend(loc='lower left')

    plt.show()          #Вывод графиков

    """model.save('C:/Users/smols/PycharmProjects/diplom/second_model')"""
