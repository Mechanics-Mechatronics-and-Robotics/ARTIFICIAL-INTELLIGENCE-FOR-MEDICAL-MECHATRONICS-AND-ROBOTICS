import os, fnmatch
import random
import numpy as np

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import utils as ut

BATCH_SIZE = 10

filenames = fnmatch.filter(os.listdir('measured_data'), '*_dis_*.txt')
frames = []

# получаем данные измерений из файлов в папке measured_data
for name in filenames:
    df = np.loadtxt('measured_data\\' + name, delimiter=' ', skiprows=1)

    # значения в столбце "#" заменяем на обозначение соответствующего класса
    data_class = name[name.find('_')+1:len(name)-4]
    class_name = 0
    if data_class == 'dis_12':
        class_name = 3
    elif data_class == 'dis_2':
        class_name = 2
    elif data_class == 'dis_1':
        class_name = 1
    df[:,0] = class_name

    frames.append(df)

# соединяем все данные в один массив
dataset = np.concatenate(frames, axis=0)
# убираем ненужные данные

# группируем и перемешиваем данные в массиве
groups = []
for i in range(0, len(dataset), BATCH_SIZE):
    temp = dataset[i,0:9]
    for j in range(1, BATCH_SIZE):
        temp = np.append(temp, dataset[j+i,1:9])
    groups.append(temp)
random.shuffle(groups)
dataset = np.array(groups)
del groups

# разделение данных на три датасета: тренировочный, тестовый, валидация
train, test = train_test_split(dataset, test_size=0.2, shuffle=False)
train, validation = train_test_split(train, test_size=0.2, shuffle=False)

# извлечение меток
train_labels = train[:,0]
train = np.delete(train, 0, axis=1)
test_labels = test[:,0]
test = np.delete(test, 0, axis=1)
validation_labels = validation[:,0]
validation = np.delete(validation, 0, axis=1)

# получение среднего и СКО из тренировочного датасета, последующая нормализация датасетов
train_mean, train_std = ut.get_mean_and_std_numpy(train)
train = ut.normalize_numpy(train, train_mean, train_std)
test = ut.normalize_numpy(test, train_mean, train_std)
validation = ut.normalize_numpy(validation, train_mean, train_std)

# Для небольших моделей использование центрального процессора предпочтительнее
# из-за высоких накладных расходов при перемещении данных в/из оперативной памяти графического ускорителя.
# Для использования графического ускорителя следует указать '/device:GPU:0'
with tf.device('/cpu:0'):
    # получение датасетов в формате, понятном для пакета tensorflow
    train_dataset = ut.df_to_dataset_numpy(train, train_labels)
    test_dataset = ut.df_to_dataset_numpy(test, test_labels)
    validation_dataset = ut.df_to_dataset_numpy(validation, validation_labels)

    # создание входного слоя
    inputs = keras.Input(shape=(80,))
    combined = inputs

    # создание промежуточных слоев
    dense = layers.Dense(56, activation="relu")
    x = dense(combined)
    x = layers.Dense(56, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    # создание выходного слоя
    outputs = layers.Dense(4, activation='softmax')(x)

    # компиляция модели, выбор оптимизатора и метрик
    model = keras.Model(inputs=inputs, outputs=outputs)    
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0005),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    
    # сохранение структуры сети в виде картинки
    keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
    
    # непосрдественно обучение нейросети
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=10)

    # проверка нейросети на тестовых данных
    loss, accuracy = model.evaluate(test_dataset)
    print("Accuracy", accuracy)
    y = model.predict(test_dataset.take(1))
    print(y)

    # сохранение модели, СКО и среднего значений
    model.save('imbalance_prediction_model.h5')
    data_file_mean_std = 'STD ' + str(train_std) + '\n' + 'Mean ' + str(train_mean)
    with open('mean_std.txt', 'w') as f:
        f.write(data_file_mean_std)
        f.close()

    # сохранение модели в формате TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('imbalance_prediction_model.tflite', 'wb') as f:
        f.write(tflite_model)
        f.close()