# defining a function to save the weights of best model
from keras.callbacks import ModelCheckpoint
from tensorflow import keras
from read_data import My_Custom_Generator
from base_model import create_model
import os
import keras.backend as K
from util import xywh2minmax,iou, yolo_head
from losscode import yolo_loss, CustomLearningRateScheduler, lr_schedule
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import motmetrics as mm

path = './MOT17/MOT17/train/'
file = os.listdir(path)
ll = []
X_train = []
Y_train = []
pic = []
for i in range(len(file)):

    if 'DPM' in file[i]:
        file2 = os.listdir(path + file[i])
        labelList = []
        picList = []
        # for j in range(len(file2)):
        gt = os.listdir(path + file[i] + '/' + file2[1])
        img = os.listdir(path+file[i] + '/' + file2[2])
        y = path + file[i] + '/' + file2[1] + '/' + gt[0]
        label = np.asarray(pd.read_csv(y, header=None), dtype='int')

        for z in range(5, len(img)-5, 1):
            ca = []
            arr = label[label[:, 0] == z]
            l = arr[arr[:, 1] < 21][:, 1:6]
            l[:, [0, 4]] = l[:, [4, 0]]

            for d in range(len(l)):
                ca.append(l[d, :])
            labelList.append(ca)

            picList.append(
                [path + file[i]+ '/' + file2[2] + '/' + img[z], path + file[i]+ '/' + file2[2] + '/' + img[z + 1],
                 path + file[i] + '/' + file2[2] + '/' + img[z+ 2], path + file[i] + '/' + file2[2] + '/'+ img[z+3],path + file[i] + '/' + file2[2] + '/'+img[z+4]])
        for k in range(len(picList)):
            X_train.append(picList[k])
            Y_train.append(labelList[k])
X_t = X_train[:int(len(X_train)*0.8)]
y_t = Y_train[:int(len(X_train)*0.8)]
X_val = X_train[int(len(X_train)*0.8):]
Y_val = Y_train[int(len(X_train)*0.8):]
batch_size = 8
my_training_batch_generator = My_Custom_Generator(X_t, y_t, batch_size)

my_validation_batch_generator = My_Custom_Generator(X_val,  Y_val, batch_size)

x_train, y_train = my_training_batch_generator.__getitem__(0)
x_val, y_val = my_training_batch_generator.__getitem__(0)

print(x_train.shape)
print(y_train.shape)

print(x_val.shape)
print(y_val.shape)
LR_SCHEDULE = [
# (epoch to start, learning rate) tuples
(0, 0.01),
(75, 0.001),
    (105, 0.0001),
]


mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')

model = create_model()
opt = keras.optimizers.Adam(lr=0.1)
model.compile(loss=yolo_loss ,optimizer=opt)
history = model.fit(x=my_training_batch_generator,
          steps_per_epoch = int(len(X_train) // batch_size),
          epochs = 1,
          verbose = 1,
          workers= 4
        )
loss = history.history['loss']
plt.plot(loss)
plt.show()
print(loss)
y_pred= model.predict(x_val)
# print(y_pred.shape)


predict_class = y_pred[..., :20]
predict_trust = y_pred[..., 20:22]
predict_box = y_pred[..., 22:]

_predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

predict_xy, predict_wh = yolo_head(_predict_box)
predict_xy = K.expand_dims(predict_xy, 4)
predict_wh = K.expand_dims(predict_wh, 4)
predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)
_predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])
predict_xy, predict_wh = yolo_head(_predict_box)

print(predict_xy.values, predict_wh)

deep_ts = y_pred
deep_acc = mm.utils.compare_to_groundtruth(y_train, deep_ts, 'iou', distth=0.5)
mh = mm.metrics.create()
metrics = ['num_frames', 'num_switches', 'idp', 'idr', 'idf1', 'mota', 'motp', 'precision', 'recall']

deep_summary = mh.compute(deep_acc, metrics=metrics, name='deepsort')

summary = pd.concat([deep_summary], axis=0, join='outer', ignore_index=False)

if os.path.exists("result.csv"):
    os.remove("result.csv")
summary.to_csv("result.csv")