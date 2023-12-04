!git clone https://github.com/HARISHREDDYCHILUMULA/Hand-Recognition.git
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D
image_size = 64
batch_size = 32
train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                        shear_range=0.2,
                                                        zoom_range=0.2,
                                                        horizontal_flip=True)
test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_set = train.flow_from_directory(r'./Hand-Recognition/train',
                                      target_size=(image_size,image_size),
                                      batch_size=batch_size,
                                      class_mode='sparse'
                                     )
test_set = test.flow_from_directory(r'./Hand-Recognition/test',
                                      target_size=(image_size,image_size),
                                    batch_size=batch_size,
                                      class_mode='sparse'
                                     )
model = Sequential([
    Conv2D(32, (3,3),activation='relu',input_shape=(image_size,image_size,3)),
    MaxPool2D(2,2),
    Conv2D(64,(3,3),activation='relu'),
    MaxPool2D(2,2),
    Conv2D(128,(3,3),activation='relu'),
    MaxPool2D(2,2),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(10,activation='softmax'),

])
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(0.001),metrics=['accuracy'])
history = model.fit(train_set,epochs=15,validation_data=test_set)
