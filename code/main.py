from functions import select_random_images_by_classes, show_images, show_one_image, distribution_chart, Pre_Process
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import sklearn
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Flatten
from sklearn.utils import shuffle

tf.disable_v2_behavior() 

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

#se pueden ver las imagenes mejor con estas funciones que encontre en internet 
"""
n_train = len(X_train)
select_random_images_by_classes(X_train, y_train, n_train)
"""
"""
#vemos algunas la distribucion del dataset
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

"""
for i in range(8):
    idx = random.randint(0, len(X_train))

    plt.figure(figsize=(1,1))
    plt.imshow(X_train[idx], cmap="gray")
    plt.show()
    plt.figure(figsize=(1,1))
    image = np.squeeze(X_train_gray[idx], axis=(2,))
    plt.imshow(image, cmap="gray")
    plt.show()
"""

EPOCHS = 15
BATCH_SIZE = 128
rate = 0.001

def LeNet(x):    
    
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x12.
    #output size = orig_size - (filter_size - 1) 
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 12), mean = mu, stddev = sigma), name="c1w") #filtro 5x5x1 con sus respectectivas pesos VER XAVIER UNIT
    conv1_b = tf.Variable(tf.zeros(12), name="c1b") #creamos bias 
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b  

    # Activation.
    conv1 = tf.nn.relu(conv1)
    
    # Pooling. Input = 28x28x12. Output = 14x14x12.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Input = 14x14x12. Output = 10x10x32.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 12, 32), mean = mu, stddev = sigma), name="c2w")
    conv2_b = tf.Variable(tf.zeros(32), name="c2b")
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x32. Output = 5x5x32.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Flatten. Input = 5x5x32. Output = 800.
    fc0 = tf.layers.flatten(conv2)
    fc0 = tf.nn.dropout(fc0, keep_prob)

    # Layer 3: Fully Connected. Input = 800. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(800, 120), mean = mu, stddev = sigma), name="fullc1w")
    fc1_b = tf.Variable(tf.zeros(120), name="fullc1b")
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84. 
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma), name="fullc2w")
    fc2_b  = tf.Variable(tf.zeros(84), name="fullc2b")
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # Layer 5: Fully Connected. Input = 84. Output = n_classes.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma), name="fullc3w")
    fc3_b  = tf.Variable(tf.zeros(43), name="fullc3b")
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)


logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
prediction_operation = tf.argmax(logits, 1)
softmax = tf.nn.softmax(logits)
top5 = tf.nn.top_k(softmax, 5)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0}) #PROBLEMA ACÁ "Cannot feed value of shape (128, 32, 32, 3) for Tensor 'Placeholder:0', which has shape '(?, 32, 32, 1)"
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")

