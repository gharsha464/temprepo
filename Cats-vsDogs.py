!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle datasets download -d salader/dogs-vs-cats
import zipfile
zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
import os
os.listdir()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train=ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)
test=ImageDataGenerator(rescale=1./255)
traindata=train.flow_from_directory(
    r'/content/dogs_vs_cats/train',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)
testdata=test.flow_from_directory(
    r'/content/dogs_vs_cats/test',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)
modell=Sequential(
    [
        Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
        MaxPooling2D(2,2),
        BatchNormalization(),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),
        Flatten(),
        Dropout(0.2),
        Dense(128,'relu'),
        Dense(32,'relu'),
        Dense(1,'sigmoid')
    ]
)
modell.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(0.001),metrics=['accuracy'])
hist=modell.fit(traindata,epochs=10,validation_data=testdata)
