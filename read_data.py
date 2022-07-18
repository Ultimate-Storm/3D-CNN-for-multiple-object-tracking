import cv2 as cv
import numpy as np

def read(image_path, label):

    img = np.zeros((5, 448, 448, 3))
    for i in range(len(image_path)):

        image = cv.imread(image_path[i])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_h, image_w = image.shape[0:2]
        image = cv.resize(image, (448, 448))
        image = image / 255.

        img[i,:,:,:] = image
    label_matrix = np.zeros([7, 7, 30])
    for l in label:

        l = np.array(l, dtype=np.int)
        xmin = l[0]
        ymin = l[1]
        xmax = l[2]
        ymax = l[3]
        cls = l[4]
        x = (xmin + xmax) / 2 / image_w
        y = (ymin + ymax) / 2 / image_h
        w = (xmax - xmin) / image_w
        h = (ymax - ymin) / image_h
        loc = [7 * x, 7 * y]
        loc_i = int(loc[1])
        loc_j = int(loc[0])
        y = loc[1] - loc_i
        x = loc[0] - loc_j
        try:
            if label_matrix[loc_i, loc_j, 24] == 0:
                label_matrix[loc_i, loc_j, cls] = cls
                label_matrix[loc_i, loc_j, 20:24] = [x, y, w, h]
                label_matrix[loc_i, loc_j, 24] = 1  # response
        except:
            pass
    return img, label_matrix


from tensorflow import keras

class My_Custom_Generator(keras.utils.Sequence) :
  
  def __init__(self, images, labels, batch_size) :
    self.images = images
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.images[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

    train_image = []
    train_label = []

    for i in range(0, len(batch_x)):
      img_path = batch_x[i]
      label = batch_y[i]
      image, label_matrix = read(img_path, label)
      train_image.append(image)
      train_label.append(label_matrix)
    return np.array(train_image), np.array(train_label)