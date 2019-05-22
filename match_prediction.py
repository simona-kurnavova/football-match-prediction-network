import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
import pandas as pd
import glob


class Network:
    def __init__(self, label_count, feature_columns):
        feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

        self.model = tf.keras.Sequential([
            feature_layer,
            #tf.keras.layers.Dense(2048, activation='relu'),
            #tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(label_count, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, train, epochs):
        self.model.fit(
            train,
            epochs=epochs,
        )

    def predict(self, data):
        return self.model.predict(data)

    def evaluate(self, data):
        return self.model.evaluate(data)


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('FTR')

    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def convert_to_number(result):
    result_str = str(result)
    if result_str == 'H':
        return 0
    if result_str == 'A':
        return 1
    return 2


if __name__ == "__main__":
    path = r'data'
    all_files = glob.glob(path + "/*.csv")
    frames = []

    for filename in all_files:
        df = pd.read_csv(filename, header=0).filter(items=['HomeTeam', 'AwayTeam', 'FTR', 'HTHG', 'HTAG', 'HTR', 'Date'])
        df['season'] = int(filename[5:7])
        frames.append(df)

    dataframe = pd.concat(frames, axis=0, ignore_index=True)
    dataframe['FTR'] = dataframe['FTR'].apply(convert_to_number)

    train, test = train_test_split(dataframe, test_size=0.2)

    print(len(train), 'train examples')
    print(len(test), 'test examples')

    feature_columns = []

    for col in ['HomeTeam', 'AwayTeam']:
        home = tf.feature_column.categorical_column_with_vocabulary_list(col, dataframe[col].unique())
        home_one_hot = tf.feature_column.indicator_column(home)
        feature_columns.append(home_one_hot)
        home_embedding = tf.feature_column.embedding_column(home, dimension=8)
        feature_columns.append(home_embedding)

    feature_columns.append(tf.feature_column.numeric_column("season"))
    #feature_columns.append(tf.feature_column.numeric_column("HTAG"))
    #feature_columns.append(tf.feature_column.numeric_column("HTHG"))

    network = Network(3, feature_columns)

    batch_size = 32
    train_ds = df_to_dataset(train, shuffle=True, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    for feature_batch, label_batch in test_ds.take(1):
        print('Every feature:', list(feature_batch.keys()))
        print('A batch of homeTeams:', feature_batch['HomeTeam'])
        print('A batch of targets:', label_batch)

    for i in range(700):
        print("EPOCH " + str(i+1))
        network.train(train_ds, 1)
        loss, accuracy = network.evaluate(test_ds)
        print("Accuracy", accuracy)

