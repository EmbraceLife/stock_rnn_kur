
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    print('yo')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        
                                                            # create stop drawing spaces or empty points for each window
        padding = [None for p in range(i * prediction_len)]
                                                            # draw from start, but very first part is empty drawing
                                                            # the visual part is real window prediction data
        plt.plot(padding + data, label='Prediction')
#         plt.legend()
    plt.show()

    
# how to load stockdata csv file
def load_data(filename, seq_len, normalise_window):
                                                        # make it a long list of numbers
    f = open(filename, 'r').read()
    data = f.split('\n')
                                                        # make it a list of lists (each list is a sequence)
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
                                                        # normalize each number
    if normalise_window:
        result = normalise_windows(result)
                                                        # turn the normalized lists into arrays
    result = np.array(result)
                                                        # take 90% of rows of data as training set
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
                                                        # shuffle the training set, and split train_x and train_y
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
                                                        # get test_x and test_y
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]
                                                        # Question: reshape train and test into 3-d dim
                                                        # (num_data_point, seq_len, 1)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]



                                                        # normalize the list of lists
def normalise_windows(window_data):
                                                        # create an empty list
    normalised_data = []
                                                        # for each element list 
    for window in window_data:
                                                        # get each number of element list, divide the first number of the 
                                                        # element list, then minus 1, save this new window in a new list
                                                        # QUESTION: why use 1st number of each window/element list, not first
                                                        # number of the entire dataset?                   
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
                                                        # store all normalized windows into a new list 
        normalised_data.append(normalised_window)
    return normalised_data



                                                        # build LSTM model to predict
def build_model(layers):                                # layers is a list [input_dim, h1_dim, h2_dim, out_dim]
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))                         # output all values of the sequence
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))                        # output last value of the sequence
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))                          # Question: kur is very alike keras, is kur just made a wrapper over keras
                                                        # kur access tensorflow through keras, not using TF code directly, right? 
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
                                                        # Predict each timestep given the last sequence of true data, 
                                                        # in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
                                                        # convert predicted from 1-d to 2-d
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

                                                        # QUESTION: I don't understand how the functions below work
                                                        # but they work, just use them first


def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(round(len(data)/prediction_len)-1)):
        
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs