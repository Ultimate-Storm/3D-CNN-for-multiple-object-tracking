from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv3D, MaxPooling3D, GlobalMaxPooling3D, Conv3D, MaxPooling3D
from tensorflow.keras.regularizers import l2
import tensorflow as tf


from tensorflow import keras
import keras.backend as K

class Yolo_Reshape(tf.keras.layers.Layer):
  def __init__(self, target_shape):
    super(Yolo_Reshape, self).__init__()
    self.target_shape = tuple(target_shape)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'target_shape': self.target_shape
    })
    return config

  def call(self, input):
    # grids 7x7
    S = [self.target_shape[0], self.target_shape[1]]
    # classes
    C = 20
    # no of bounding boxes per grid
    B = 2

    idx1 = S[0] * S[1] * C
    idx2 = idx1 + S[0] * S[1] * B
    
    # class probabilities
    class_probs = K.reshape(input[:, :idx1], (K.shape(input)[0],) + tuple([S[0], S[1], C]))
    class_probs = K.softmax(class_probs)

    #confidence
    confs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0],) + tuple([S[0], S[1], B]))
    confs = K.sigmoid(confs)

    # boxes
    boxes = K.reshape(input[:, idx2:], (K.shape(input)[0],) + tuple([S[0], S[1], B * 4]))
    boxes = K.sigmoid(boxes)

    outputs = K.concatenate([class_probs, confs, boxes])
    return outputs

def create_model():
	lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)

	nb_boxes=1
	grid_w=7
	grid_h=7
	cell_w=64
	cell_h=64
	img_w=grid_w*cell_w
	img_h=grid_h*cell_h

	model = Sequential()
	model.add(Conv3D(filters=64, kernel_size= (1, 7, 7), strides=(1, 1, 1), input_shape =(5, img_h, img_w, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding = 'same'))

	model.add(Conv3D(filters=192, kernel_size= (3, 3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding = 'same'))

	model.add(Conv3D(filters=128, kernel_size= (1, 1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	model.add(Conv3D(filters=256, kernel_size= (3, 3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	model.add(Conv3D(filters=256, kernel_size= (1, 1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	model.add(Conv3D(filters=512, kernel_size= (3, 3,3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding = 'same'))

	model.add(Conv3D(filters=256, kernel_size= (1, 1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	model.add(Conv3D(filters=512, kernel_size= (3, 3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	model.add(Conv3D(filters=256, kernel_size= (1, 1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	model.add(Conv3D(filters=512, kernel_size= (3, 3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	model.add(Conv3D(filters=256, kernel_size= (1, 1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	model.add(Conv3D(filters=512, kernel_size= (3, 3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	model.add(Conv3D(filters=256, kernel_size= (1, 1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	model.add(Conv3D(filters=512, kernel_size= (3, 3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	model.add(Conv3D(filters=512, kernel_size= (1, 1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	model.add(Conv3D(filters=1024, kernel_size= (3, 3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding = 'same'))

	model.add(Conv3D(filters=512, kernel_size= (1, 1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	model.add(Conv3D(filters=1024, kernel_size= (3, 3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	model.add(Conv3D(filters=512, kernel_size= (1, 1, 1), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	# model.add(Conv3D(filters=1024, kernel_size= (3, 3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	# model.add(Conv3D(filters=1024, kernel_size= (3, 3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	# model.add(Conv3D(filters=1024, kernel_size= (3, 3, 3), strides=(2, 2, 2), padding = 'same'))
	#
	# model.add(Conv3D(filters=1024, kernel_size= (3, 3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))
	# model.add(Conv3D(filters=1024, kernel_size= (3, 3, 3), padding = 'same', activation=lrelu, kernel_regularizer=l2(5e-4)))

	model.add(Flatten())
	model.add(Dense(512))
	# model.add(Dense(1024))
	model.add(Dropout(0.5))
	model.add(Dense(1470, activation='sigmoid'))
	model.add(Yolo_Reshape(target_shape=(7,7,30)))
	model.summary()
	return model

