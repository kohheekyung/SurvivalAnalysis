import os
import sys
import pickle
import glob
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from keras.initializers import glorot_uniform
from keras.models import Sequential,Model
from keras.layers import Dense, Activation, Dropout, ActivityRegularization, Input
from keras.optimizers import SGD, RMSprop, Adam
from keras.regularizers import l2
from keras.backend import tensorflow_backend as K
import tensorflow as tf
from matplotlib import pyplot as plt

from lifelines.utils import concordance_index
from lifelines import CoxPHFitter


# Multi labeling string values
def multi_encoding(df):

    one_hot_df = pd.get_dummies(df)
    # print(one_hot_df.columns)

    for i in range(len(one_hot_df.columns)):
        label = one_hot_df.columns[i]

        df = df.apply(lambda x: i if x == label else x)
        print('convert', label, 'to', i)

    return df, one_hot_df.columns


def standardize(df, mean, std):
    df = (df - mean) / std
    return df

def preprocessing(df, mean, std):
    data_event = df['CONVERTER']
    data_time = df['CONV_TIME']

    data_event, event_col = multi_encoding(data_event)
    data_event = data_event.apply(lambda x: 1 if x == 0 else x)
    # Time labeling
    data_time, time_col = multi_encoding(data_time)

    #drop unnecessary feature
    data = df.drop(['CONVERTER', 'CONV_TIME', 'RID'],axis=1).copy()

    #drop features for check feature importance
    # data = data.drop(['EcogPtLang', 'EcogPtLang_bl', 'EcogPtMem', 'EcogPtMem_bl'],
    #                axis=1).copy()

    # standarlize features only num value (not class values)
    string_vars = ['DX_bl', 'PTGENDER', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'DX']
    string_data_x = df[string_vars]
    data_x = data.drop(string_vars, axis=1)
    data_x = standardize(data_x, mean, std)
    data_x[string_vars] = string_data_x

    # print(data_x[:90], data_event[:90], data_time[:90])
    # return feature, data occurence, dataa occur time, data feature column name , time column name
    return np.array(data_x), np.array(data_event), np.array(data_time), data_x.columns, time_col


def build_model(n_features):

    inp = Input(shape=(n_features,))

    x1 = Dense(32, init='glorot_uniform', activation='relu')(inp)
    x2 = Dropout(0.3)(x1)
    x3 = Dense(32, init='glorot_uniform', activation='relu')(x2)
    x4 = Dropout(0.2)(x3)
    x = Dense(32, init='glorot_uniform', activation='relu')(x4)
    # classification to 14 class ( Month 6 ~ 244)
    out = Dense(14, activation='sigmoid')(x)
    model = Model(inp, out)

    return model

# load data
data = pd.read_csv('./data/2020010629_Imputed.csv')

#get mean, std except the class-valued feature
string_vars = ['DX_bl', 'PTGENDER', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'DX', 'CONVERTER','CONV_TIME', 'RID']
mean = data.drop(string_vars, axis =1).mean(axis=0)
std = data.drop(string_vars, axis =1).std(axis=0)

#preprocess & standardize except the class-valued feature
x_data, e_data, t_data, n_features, time_col= preprocessing(data, mean, std)

#sort datas in descending order
sort_idx = np.argsort(t_data)[::-1]
x_data = x_data[sort_idx]
e_data = e_data[sort_idx]
t_data = t_data[sort_idx]

#time data to multi class
t_data = t_data.reshape(-1, 1)
enc =OneHotEncoder()
enc.fit(t_data)
print(t_data[0])
t_data = enc.transform(t_data).toarray()

#t_data = t_data[:,:,np.newaxis]
#t_data = t_data.reshape(1, -1)

# split data train 0.8 test 0.2
split_idx = int(x_data.shape[0] - (x_data.shape[0] * 0.2))
x_train = x_data[:split_idx]
e_train = e_data[:split_idx]
t_train = t_data[:split_idx]

x_test = x_data[split_idx:]
e_test = e_data[split_idx:]
t_test = t_data[split_idx:]

print(x_train.shape)
print(t_train.shape)
print(e_train.shape)

# Model training
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

model = build_model(n_features)
opt = Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(x_train, t_train, epochs=15, shuffle=False, validation_split= 0.2 )
# Save history plt

fig, ax = plt.subplots(1, 1, figsize=[5, 5])
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')

plt.legend(loc='upper right')
ax.set_xlabel("No. epochs")
ax.set_ylabel("Loss")
plt.savefig('./history_mlp')

# Model evaluation with test data
surv = np.array(model.predict(x_test))
scores = model.evaluate(x_test, t_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Plot Survival func from test data
def softmax(x):
    exp = np.exp(x)
    return exp / exp.sum(1, keepdims=True)

surv = softmax(surv)
surv = 1 - surv.cumsum(1)

surv_mean = np.mean(surv, axis = 0)
surv_std = np.std(surv, axis = 0)

lw = surv_mean- surv_std
up = surv_mean+ surv_std

surv = pd.DataFrame(surv_mean.transpose())
surv_std = pd.DataFrame(surv_std.transpose())
print(time_col)
print(surv.index.values * 6)
surv.index= surv.index.values * 6
surv.tail()
surv.plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Month')
plt.savefig('./surv_func_mlp')



