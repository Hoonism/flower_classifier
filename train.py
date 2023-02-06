import cv2
import matplotlib.pyplot as plt
import matplotlib.image as im
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import VGG19

vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(170, 170, 3))
for layer in vgg_model.layers[:17]:
    layer.trainable = False
gpus = tf.config.experimental.list_physical_devices('GPU')


img_array = []
for file, root, direcs in os.walk("dataset"):
    i = 0
    for direc in direcs:
        try:
            filename = f"dataset/{direc}"
            img = cv2.imread(filename)
            new_img = cv2.resize(img, (170, 170))
            if direc.split("_")[1] == "1":
                label = 0
            if direc.split("_")[1] == "2":
                label = 1
            if direc.split("_")[1] == "3":
                label = 2
            if direc.split("_")[1] == "4":
                label = 3
            img_array.append([new_img, label])
            i += 1
            if i % 10 == 0:
                print(i)
        except:
            pass


random.shuffle(img_array)
train_x = []
train_y = []
for img, label in img_array:
    train_x.append(img)
    train_y.append(label)

train_x = np.array(train_x)
train_y = np.array(train_y)
train_x = train_x / 255.0

train_x = train_x.reshape((-1, 170, 170, 3))
print(train_x.shape, train_y.shape)

model = vgg_model.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = BatchNormalization()(model)
model = Dense(1024, activation='relu')(model)
model = Dense(4, activation='softmax')(model)

final_model = Model(inputs=vgg_model.input, outputs=model)
# model.add(Conv2D(256, (3, 3), input_shape=train_x.shape[1:], activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.4))
# model.add(BatchNormalization())

# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.4))
# model.add(BatchNormalization())

# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.4))
# model.add(BatchNormalization())


# model.add(Flatten())

# model.add(Dense(512, activation='relu'))
# model.add(Dense(4, activation='softmax'))

filepath = 'temp/best_model_pretrained.epoch{epoch:02d}-loss{val_loss:.2f}.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
print(final_model.summary())

final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
final_model.fit(train_x, train_y, validation_split=0.15, batch_size=16, epochs=16, callbacks=[model_checkpoint_callback])
final_model.save("logs/mask_detect_pretrained")

