import numpy as np
import pandas as pd
import tensorflow as tf

# convert data into dataset from pandas
def df_to_dataset(dataframe, batch_size):
    dataframe = dataframe.copy()
    labels = dataframe.pop('class')
    dataset = tf.data.Dataset.from_tensor_slices((dataframe.values, labels.values))
    dataset = dataset.batch(batch_size)
    return dataset

#convert data into dataset from numpy array
def df_to_dataset_numpy(dataframe, labels):
    dataset = tf.data.Dataset.from_tensor_slices((dataframe, labels))
    dataset = dataset.batch(1)
    return dataset

def get_mean_and_std_numpy(data):
    means = data.mean(axis = 0).mean(axis = 0)
    std = np.sqrt(np.power(data - means, 2).sum(axis = 0).sum(axis = 0) / (data.shape[0] * data.shape[1] - 1))
    return means, std

def normalize_numpy(data, means, std):
    return (data - means) / std


def normalize(in_df, mean, std):
    data = in_df.copy()
    temp_col = data['class']
    del data['class']
    norm_data = normalize_numpy(data.values, mean, std)
    output_df = pd.DataFrame(norm_data, columns=data.columns)
    output_df = output_df.reset_index(drop=True)
    temp_col = temp_col.reset_index(drop=True)
    return output_df.join(temp_col)

def get_mean_and_std(in_df):
    data = in_df.copy()
    temp_col = data['class']
    del data['class']
    return means_and_std_signal_numpy(data.values)