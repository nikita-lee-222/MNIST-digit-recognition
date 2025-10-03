import tensorflow as tf
from tensorflow import keras

#Model initialization
model = keras.models.load_model("../model/mnist_model.h5")

#Data initialization
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#Resize
x_test = tf.image.resize(x_test[..., None], (280, 280)).numpy()
x_test = x_test.astype("float32") / 255.0

y_test = keras.utils.to_categorical(y_test, 10)

#Accuracy
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Accuracy: {acc:.4f}")
