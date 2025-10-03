import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image

#MNIST initialization
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#Generator with augmentation
def generator(x, y, batch_size=8):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    while True:
        idx = np.random.randint(0, x.shape[0], batch_size)
        batch_x = np.array([
            np.array(
                Image.fromarray(x[i]).resize((280,280))
            ).reshape(280,280,1) for i in idx
        ]).astype('float32') / 255.0
        batch_y = y[idx]
        yield batch_x, batch_y

train_gen = generator(x_train, y_train, batch_size=8)
val_gen = generator(x_test, y_test, batch_size=8)

#Conv2d settings
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(280,280,1)),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#We use CPU for training. Optionally you can use GPU, but check if it is NVidia. Tensorflow does not support AMD GPUs.
model.fit(train_gen,
          steps_per_epoch=len(x_train)//8,
          validation_data=val_gen,
          validation_steps=len(x_test)//8,
          epochs=10,
          callbacks=[early_stop])

#Modedl saving
model.save("../model/mnist_model.h5")
print("Model was saved like mnist_model.h5")
