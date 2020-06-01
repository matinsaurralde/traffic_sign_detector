from functions import select_random_images_by_classes, show_images, show_one_image, distribution_chart
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn
import tensorflow as tf


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




    


