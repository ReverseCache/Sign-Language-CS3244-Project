import pandas as pd
import cv2 as ocv
import pickle
import os
import numpy as np
from itertools import zip_longest
from tensorflow import keras
import tensorflow as tf
import kerasncp as kncp
from kerasncp import wirings
from kerasncp.tf import LTCCell
import keras.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping

# img = ocv.imread('.png'.format(id))
# img = ocv.resize(img, (640, 640))
height = 640
width = 640
data_directory = r''
video_frames_directory = data_directory + ''
min_frames = 30
data_payload = {}
name_dict = {}
counter = 0
#global data_directory
print(data_directory)
train_x = []
train_y = []
test_x = []
test_y = []
'''
for file in os.listdir(data_directory):
    if file.endswith(".pkl"):
        file_directory = os.path.join(data_directory, file)
        scene = pd.read_pickle(file_directory)
        windows = []
        for window in scene:
          if len(window) < min_frames:
            continue
          else:
              limit = 0
              windowx = []
              for frame in window:
                if frame[3] < 0: #datapoint there = 0
                  continue
                else:
                  windowx.append(frame[:126])
                  limit+=1
                  if limit == min_frames:
                    break
          if len(windowx)<min_frames:
            continue
          else:
            windows.append(windowx) 
        if len(windows) < 10:
          continue
        else:
          sep = '_'
          stripped_file_name = file.split(sep, 1)[0]
          name_dict[counter] = stripped_file_name
          #train, test = windows[:int(len(windows)*0.8)] , windows[int(len(windows)*0.8):]
          train, test = windows, windows
          train_x.extend(train)
          test_x.extend(test)
          train_y.extend([counter]*len(train))
          test_y.extend([counter]*len(test))
          data_payload[stripped_file_name] = counter
          counter +=1
'''
# for x in train_x:
#   print(len(x))
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)
x_train = train_x
y_train = tf.keras.utils.to_categorical(train_y).astype(int)
y_train.shape
np.nan_to_num(x_train)
img_size = (height, width)


def s_map(model, image):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = model(image)
    gradient = np.array(tf.reduce_max(tape.gradient(loss, image), axis=-1))
    saliency_map = (gradient - np.min(gradient)) / \
        (np.max(gradient) - np.min(gradient) + keras.backend.epsilon())
    return saliency_map


early_stopping = EarlyStopping(
    monitor='val_loss', patience=3, min_delta=0.0001, mode='max')
sample_input = (min_frames, 126)
model = keras.models.Sequential(
    [
        keras.layers.LSTM(64, return_sequences=True,
                          activation='relu', input_shape=sample_input),
        keras.layers.LSTM(128, return_sequences=True, activation='relu'),
        keras.layers.LSTM(64, return_sequences=False, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(len(data_payload),
                           activation='softmax', name='output')
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(0.1),
    loss='categorical_crossentropy',
    metrics=[
        # metrics.MeanSquaredError(),
        # metrics.AUC(),
        metrics.Accuracy(),
        metrics.AUC(),

    ]
)
model.summary()
model.fit(x_train, y_train, validation_split=0.1, verbose=1, epochs=200, steps_per_epoch=1,
          batch_size=1, callbacks=[early_stopping])  # ,callbacks=[tb_callback]) #batch_size=1, verbose=1)
res = model.predict(test_x)
multilabel_confusion_matrix(train_y, res)
for frame in os.listdir(video_frames_directory):
    if frame.endswith(".png"):
        s_map(model, frame)
wiring = kncp.wirings.FullyConnected(8, 1)
ltc_cell = LTCCell(wiring)
model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=(None, 2)),
        keras.layers.RNN(ltc_cell, return_sequences=True),
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(0.01), loss='mean_squared_error'
)
height, width, channels = (height, width, min_frames)
ncp_wiring = kncp.wirings.NCP(
    inter_neurons=12,
    command_neurons=8,
    motor_neurons=1,
    sensory_fanout=4,
    inter_fanout=4,
    recurrent_command_synapses=4,
    motor_fanin=6,
)
ltc_cell = LTCCell(wiring)
#img_path = keras.utils.get_file(r'')
model = keras.models.Sequential(
    [
        keras.layers.InputLayer(input_shape=sample_input),
        # keras.layers.TimeDistributed(
        #     keras.layers.Conv2D(32, (5, 5), activation="relu")
        # ),
        # keras.layers.TimeDistributed(keras.layers.MaxPool2D()),
        # keras.layers.TimeDistributed(
        #     keras.layers.Conv2D(64, (5, 5), activation="relu")
        # ),
        # keras.layers.TimeDistributed(keras.layers.MaxPool2D()),
        # keras.layers.TimeDistributed(keras.layers.Flatten()),
        keras.layers.TimeDistributed(
            keras.layers.Dense(64, activation="relu")),
        keras.layers.RNN(ltc_cell, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Activation("softmax")),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(len(data_payload),
                           activation='softmax', name='output')
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(0.1),
    loss='categorical_crossentropy',
    metrics=[
        # metrics.MeanSquaredError(),
        # metrics.AUC(),
        metrics.Accuracy(),
        metrics.AUC(),

    ]
)
model.summary()
sns.set_style("white")
plt.figure(figsize=(12, 12))
legend_handles = ltc_cell.draw_graph(
    layout='spiral', neuron_colors={"command": "tab:cyan"})
plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()
model.fit(x_train, y_train, validation_split=0.1, verbose=1, epochs=200, steps_per_epoch=1,
          batch_size=1, callbacks=[early_stopping])  # ,callbacks=[tb_callback]) #batch_size=1, verbose=1)
res = model.predict(test_x)
multilabel_confusion_matrix(train_y, res)
for frame in os.listdir(video_frames_directory):
    if frame.endswith(".png"):
        s_map(model, frame)
model = keras.models.Sequential(
    [
        keras.layers.InputLayer(input_shape=sample_input),
        #keras.layers.GRU(128, return_sequences=True),
        keras.layers.SimpleRNN(64),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(len(data_payload),
                           activation='softmax', name='output')
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(0.1),
    loss='categorical_crossentropy',
    metrics=[
        # metrics.MeanSquaredError(),
        # metrics.AUC(),
        metrics.Accuracy(),
        metrics.AUC(),

    ]
)
model.summary()
model.fit(x_train, y_train, validation_split=0.1, verbose=1, epochs=200, steps_per_epoch=1,
          batch_size=1, callbacks=[early_stopping])  # ,callbacks=[tb_callback]) #batch_size=1, verbose=1)
res = model.predict(test_x)
multilabel_confusion_matrix(train_y, res)
