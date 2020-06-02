from functions import select_random_images_by_classes, show_images, show_one_image, distribution_chart, Pre_Process
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn
import tensorflow as tf
from tensorflow.keras.layers import Flatten



training_file = '/Users/matinsaurralde/Documents/Beca Tarpuy/Ingenia/3er_nivel/traffic_sign/traffic-signs-data/train.p'
validation_file= '/Users/matinsaurralde/Documents/Beca Tarpuy/Ingenia/3er_nivel/traffic_sign/traffic-signs-data/valid.p'
testing_file = '/Users/matinsaurralde/Documents/Beca Tarpuy/Ingenia/3er_nivel/traffic_sign/traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

#verificamos que num de images = num labels en cada set
assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))

#printeamos info 
print("Tamanio imagen:" + str(X_train[0].shape))
print("Cantidad imagenes de Train:" + str(len(X_train)))
print("Cantidad imagenes de Validacion:" + str(len(X_valid)))
print("Cantidad imagenes de Test:" + str(len(X_test)))
classes = []
for i in y_train:
    if (i not in classes):
        classes.append(i)
num_classes = len(classes)
print("Cantidad de clases distintas: " + str(num_classes))
#tambien se podria hacer asi -> print(len(np.unique(y_valid o y_train)))

#vemos una imagen en el indice que queremos
#show_one_image(X_train, 2900)

"""se pueden ver las imagenes mejor con estas funciones que encontre en internet 
n_train = len(X_train)
select_random_images_by_classes(X_train, y_train, n_train)
"""

"""vemos algunas la distribucion del dataset
num_classes, counts = np.unique(y_train, return_counts=True)
distribution_chart(num_classes, counts, 'Classes', 'Training Examples')
"""

#shuffleamos la data pq puede arruinar el train

pre_process = Pre_Process ()
normalized_dataset = pre_process.normalize_dataset(X_train)
print(pre_process.get_mean_dataset(X_train))
print(pre_process.get_mean_dataset(normalized_dataset))
shufled = pre_process.shuffle_dataset(normalized_dataset,y_train)
X_train_gray = pre_process.dataset2gray(X_train)
X_valid_gray = pre_process.dataset2gray(X_valid)
X_test_gray = pre_process.dataset2gray(X_test)
print(pre_process.get_mean_dataset(X_train_gray))
X_train_gray = pre_process.normalize_dataset(X_train_gray)
X_valid_gray = pre_process.normalize_dataset(X_valid_gray)
X_test_gray  = pre_process.normalize_dataset(X_test_gray)
print(pre_process.get_mean_dataset(X_train_gray))
shufled_gray = pre_process.shuffle_dataset(X_train_gray,y_train)
print (shufled_gray[0][0].shape)


for i in range(8):
    idx = random.randint(0, len(X_train))

    plt.figure(figsize=(1,1))
    plt.imshow(X_train[idx], cmap="gray")
    plt.show()
    plt.figure(figsize=(1,1))
    image = np.squeeze(X_train_gray[idx], axis=(2,))
    plt.imshow(image, cmap="gray")
    plt.show()

    


