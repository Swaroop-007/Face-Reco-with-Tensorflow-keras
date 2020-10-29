
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

#PART 1 - DATA PREPROCESSING
"""Preprocessing Training dataset"""
train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

train_gen=train_datagen.flow_from_directory('dataset/training_set',
                                            target_size=(64,64),
                                            batch_size=15,
                                            class_mode='binary')
"""Preprocessing test dataset"""
test_datagen=ImageDataGenerator(rescale=1./255)
valid_data=test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64,64),
                                            batch_size=15,
                                            class_mode='binary')
#PART 2 BUILDING THE CNN
"""Convolution"""
cnn=tf.keras.models.Sequential()

""" Adding 1st layer"""
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, 
                               activation='relu', 
                               input_shape=[64,64,3]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

""" Adding 2nd Layer"""
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

""" Adding 3rd Layer"""
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

""" Adding 3rd Layer"""
cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

"""Flattening"""
cnn.add(tf.keras.layers.Flatten())

""" Full Connection """
cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))

#cnn.add(tf.keras.layers.Dropout(0.5))

""" O/P Layer """
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


#Part 3 Training the CNN
"""Compiling the cnn"""
cnn.compile(optimizer='adam', 
            loss='binary_crossentropy',
            metrics=['accuracy'])

""" Training the CNN on training set and evaluating on test_est"""
cnn.fit(x=train_gen, validation_data=valid_data, epochs=15)

# Part 4 Making the prediction

test_img=tf.keras.preprocessing.image.load_img('Swaroop.jpg',
                                               target_size=(64,64))
plt.imshow(test_img)
test_img=tf.keras.preprocessing.image.img_to_array(test_img)
test_img=np.expand_dims(test_img, axis=0)

result=cnn.predict(test_img)
train_gen.class_indices
if result[0][0]==1:
    prediction='Swaroop'
else:
    prediction='Hugh Jackman'
print(prediction)