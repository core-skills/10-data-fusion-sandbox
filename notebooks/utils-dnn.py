
#
# Some functions to be used in the tutorial
#
# Developed by Debora Cristina Correa

import datetime
import pandas as pd
import matplotlib.pyplot as plt # for 2D plotting
import numpy as np
import seaborn as sns # plot nicely =)

from sklearn.base import clone

from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve

import keras

def create_sin_data(samples=5000, period=10):
    '''
    Creates sin wave data.

    samples: number of samples
    period: length of one cycle of the curve
    '''
    # creating the sampling space
    x = np.linspace(-period * np.pi, period * np.pi, samples) 

    # Create the sin data and store it in a Dataframe format
    series = pd.DataFrame(np.sin(x))
    
    return series

def create_window_sin(data, window_size = 50, drop_nan = False):
    '''
    Samples the data by using move window sliding

    data: sampled data in a DataFrame format
    window_size: moving window size
    drop_nan: remove the missing values (NaN)
    '''
    data_bk = data.copy()
    for i in range(window_size):
        data = pd.concat([data, data_bk.shift(-(i + 1))], 
                            axis = 1)
    
    # if drop_nan is true, we remove the NaN (Not a Number) values
    if drop_nan:
        data.dropna(axis=0, inplace=True)
        
    return data

def train_test_split( data, train_size =0.8):
    '''
    data: windowed dataset in DataFrame format
    train_size: size of the training dataset
    '''

    nrow = round(train_size * data.shape[0])

    # iloc allows the using of slicing operation and returns
    # the related DataFrame. Note that, this is different of using 
    # data.values, in which the returned elements are numpy.array
    train = data.iloc[:nrow, :] # train dataset
    test = data.iloc[nrow:, :]  # test dataset

    train_X = train.iloc[:, :-1]
    test_X = test.iloc[:, :-1]

    train_Y = train.iloc[:, -1]
    test_Y = test.iloc[:, -1]

    return train_X, train_Y, test_X, test_Y

def plot_training_test_data( data, train_size ):

    nrow = round(train_size * data.shape[0])

    # iloc allows the using of slicing operation and returns
    # the related DataFrame. Note that, this is different of using 
    # data.values, in which the returned elements are numpy.array

    f, (ax1) = plt.subplots(1,1, figsize = (15, 5))

    data.iloc[:nrow].plot(ax=ax1, color='r')
    data.iloc[nrow-1:].plot(ax=ax1, color='g')

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Training and test data")

    ax1.legend(['Train dataset', 'Test dataset'], loc='best')

def plot_samples(data, labels, class_name, n_samples=5):
    
    for cls_id, cls in zip(np.unique(labels), class_name):
        images = data[labels == cls_id]
        print('____________________________________________________________________________________________')
        print('\n{} - number of samples: {}'.format(cls,len(images)))
        
        ridx = np.random.randint(0, images.shape[0], n_samples)
        
        i = 0
        himg = []
        for col in range(1,n_samples+1):
            if len(himg) == 0:
                himg = images[ridx[i],:,:,:]
            else:
                himg = np.hstack((himg, images[ridx[i],:,:,:]))
            i += 1


        plt.figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.axis('off')
        plt.imshow(himg)
        plt.show()

def plot_loss_and_accuracy(history):
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(1, 2, figsize=(14, 4), sharex=True)
    sns.despine(left=True)
    
    history = history.history    
    
    # Loss
    ax[0].plot(range(1,len(history['loss'])+1),history['loss'])
    ax[0].plot(range(1,len(history['val_loss'])+1),history['val_loss'])
    ax[0].set_title('Model Loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train set', 'Dev set'], loc='best')
    
    # Accuracy
    ax[1].plot(range(1,len(history['binary_accuracy'])+1),history['binary_accuracy'])
    ax[1].plot(range(1,len(history['val_binary_accuracy'])+1),history['val_binary_accuracy'])
    ax[1].set_title('Model Accuracy')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train set', 'Dev set'], loc='best') 
    
    plt.tight_layout()
    plt.show()

def plot_loss_and_accuracy_am2(history):
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(1, 2, figsize=(14, 4), sharex=True)
    sns.despine(left=True)
    
    history = history.history    
    
    # Loss
    ax[0].plot(range(1,len(history['loss'])+1),history['loss'])
    ax[0].plot(range(1,len(history['val_loss'])+1),history['val_loss'])
    ax[0].set_title('Model Loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train set', 'Dev set'], loc='best')
    
    # Accuracy
    ax[1].plot(range(1,len(history['acc'])+1),history['acc'])
    ax[1].plot(range(1,len(history['val_acc'])+1),history['val_acc'])
    ax[1].set_title('Model Accuracy')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train set', 'Dev set'], loc='best') 
    
    plt.tight_layout()
    plt.show()
    
def create_window(data, n_in = 1, n_out = 1, drop_nan = False):
    '''
    Converts the time-series to a supervised learning problem
    Based on: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/

    data: Sequence of observations as a list or 2D NumPy array.
    n_in: number of lag observations as input (X). 
         Values may be between [1..len(data)] Optional. Defaults to 1.
    n_out: number of observations as output (y). 
           Values may be between [0..len(data)-1]. Optional. Defaults to 1.
    drop_nan: boolean whether or not to drop rows with NaN values. Optional. 
              (Defaults to False).
    '''
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if drop_nan:
        agg.dropna(inplace=True)
    return agg

def plot_loss(history):
    
    history = history.history
    
    # Loss
    plt.plot(range(1,len(history['loss'])+1),history['loss'])
    plt.plot(range(1,len(history['val_loss'])+1),history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train set', 'Dev set'], loc='best')
    
    plt.show()

def inverse_transform_multiple(test_X, test_y, yhat, scaler, n_hours, n_features):

    rtest_X = test_X.reshape((test_X.shape[0], n_hours*n_features))

    # invert scaling for forecast
    inv_yhat = np.concatenate((rtest_X[:, (n_hours-1)*n_features:-1], yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,n_features-1]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((rtest_X[:, (n_hours-1)*n_features:-1], test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,n_features-1]

    return inv_y, inv_yhat

def plot_comparison(series, series_label, title):
    '''
    Plot two time series, both are numpy.arrays
    '''

    plt.figure(figsize = (15, 5))

    for i in range(len(series)):
        plt.plot(series[i], label=series_label[i])
    
    plt.xlabel("x")
    plt.ylabel("Silica Concentrate")
    plt.title(title)
    plt.legend(loc="upper right")

    plt.show()
