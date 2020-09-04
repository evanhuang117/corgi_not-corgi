import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import imghdr
import cv2


def main():
    data_dir = pathlib.Path("./corgi_images2")
    image_count = len(list(data_dir.glob('*.jpg')))
    print(image_count)

    batch_size = 32
    img_height = 180
    img_width = 180
    # CLASS NAMES
    class_names = ["corgi", "not_corgi"]
    print(class_names)

    # CREATE DATASETS
    colormode = "rgb"
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        labels="inferred",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode=colormode)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        labels="inferred",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode=colormode)

    # CREATE TEST DATASET
    test_dir = pathlib.Path("./test")
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode=colormode)

    #CACHE IMAGES
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # CREATE MODEL
    num_classes = 2

    model = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    """
    model = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    """
    epochs = 38
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    filename = "corgi_not-corgi epochs-" + str(epochs)
    model.save("./models/" + filename)

    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_ds)

    #SHOW PREDICTIONS
    print(predictions)
    print(test_ds.take(1))
    images, labels = next(iter(test_ds.take(1)))
    labels = labels.numpy()
    rows = 3
    cols = 3
    figure = plot_predictions(rows, cols, predictions, labels, images, class_names)
    plot_val_accuracy(hist, figure)
    plt.tight_layout()
    plt.show()

def plot_val_accuracy(hist, fig):
    print(hist.history)
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']
    train_loss = hist.history['loss']
    plt.subplot(3, 6, (13,15))
    plt.plot(range(len(val_loss)), val_loss, label="validation loss")
    plt.plot(range(len(train_loss)), train_loss, label="training loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.subplot(3, 6, (16, 18))
    plt.plot(range(len(val_acc)), val_acc, label="max acc" + str(np.argmax(val_acc)) + "epochs")

    plt.xlabel("epochs")
    plt.ylabel("%")
    plt.legend()

def plot_predictions(num_rows, num_cols, predictions, labels, images, class_names):
    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_images = 6
    #num_images = num_rows * num_cols
    figure = plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], labels, images, class_names)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], labels)
    return figure

def plot_image(i, predictions_array, true_label, img, class_names):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img.numpy().astype("uint8"))
    #plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(2))
    plt.yticks([])
    thisplot = plt.bar(range(2), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

if __name__ == "__main__":
    main()
