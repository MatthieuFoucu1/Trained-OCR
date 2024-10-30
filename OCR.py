#---imports---
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
#removed a warning
os.environ['TK_SILENCE_DEPRECATION'] = '1'
import numpy as np
import matplotlib.pyplot as plt
import cv2
from draw import DrawScreen
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import tkinter as tk


#------Functions-------
root = tk.Tk()
root.withdraw()

# get image from drawing
def draw():
    screen = DrawScreen(root)
    root.deiconify()  # Make the window visible
    root.mainloop()   # Keep the image loaded
    return screen.get_image()


def imageProccess(image): # gray - scale - center - expand - normalize - invert

    #binary Thresholding is optional since my images already come in black and white

    # Convert to grayscale first - since scaling is better on grayscaled
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
    # Scale to 20x20 while preserving aspect ratio
    h, w = gray.shape
    # this allows it to work for any shape of drawing board
    scale = min(20/h, 20/w)
    new_h, new_w = int(h * scale), int(w * scale)
    scaled_image = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a 28x28 white canvas to put the 20x20 centered on it
    canvas = np.ones((28, 28), dtype=np.uint8) * 255
    
    # Calculate center
    offset_y = (28 - new_h) // 2
    offset_x = (28 - new_w) // 2
    
    # Place the scaled image onto the canvas
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = scaled_image

    # Optional Apply thresholding to make it binary after
    # thresh_hold, binary_image = cv2.threshold(canvas, 128, 255, cv2.THRESH_BINARY)

    # Normalize the image
    normalized_image = canvas.astype('float32') / 255.0
    
    # Invert the image
    inverted = 1 - normalized_image
    
    return inverted

# useful to see the drawn image vs the image the computer reads after proccessing
def compare(image, p_image):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Blurred and Resized Image (28x28)')
    plt.imshow(p_image, cmap='gray')
    plt.axis('off')
    plt.show()
    
def getSummary(model):
    # prints information about each layer shape
    model.summary()
    # gets the parameters, aka the weights that the model has used to apply to each layer and bais
    total_params = model.count_params()
    print(f"Total number of parameters: {total_params}")  


# nueral network with tensor flow
def neuralNetwork():
    #load in the dataset of 28x28 images of numbers
    mnist = tf.keras.datasets.mnist
  # (image,    label ), (image,  label )
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # make sure the depth is all the same
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # normalize the pixels from the images/ black or white
    x_train, x_test = x_train / 255.0, x_test / 255.0   
    
    #build model CNN: sequental - good for adding layers on top of one another
    model = models.Sequential()
    # add first layer of (#kernals, size of filter, Rectified Linear Unit, shape of data)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    # downsizes the computations needed, in each 2x2 the max value is kept
    model.add(layers.MaxPooling2D((2, 2)))
    # add another layer this time increase the filters
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # turn all the data into a 1d array
    model.add(layers.Flatten())
    # a dense layer with 64 neurons 
    model.add(layers.Dense(64, activation='relu'))
    # drops half the neurons it reduces the chances of overfitting by
    model.add(layers.Dropout(0.5))
    # last layer of 10 neurons to assure that it outputs 0-9 
    model.add(layers.Dense(10, activation='softmax'))


    # get a summary of the model
    getSummary(model)

    # compile the model with defualt parameters
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #train the model
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc}')

    return model

def train_or_load_model():
  # if model already exists, use it, if not train and save it
    if not os.path.exists('mnist_model.keras'):
        print("Training new model...")
        model = neuralNetwork()
        model.save('mnist_model.keras')

    else:
        print("Loading existing model...")
        model = tf.keras.models.load_model('mnist_model.keras')

    return model

def run_gui():
  #separate function to load the drawing board
    image = draw()
    processed_image = imageProccess(image)
    # compare(image, processed_image)
    return processed_image


# ------Main------  
def main():
    model = train_or_load_model()
    # was getting errors when first writing, should be good now
    try:
        processed_image = run_gui()
        prediction = model.predict(processed_image.reshape(1, 28, 28, 1))
        predicted_digit = np.argmax(prediction)

      # the output of the CNN
        print(f"The predicted digit is: {predicted_digit}")
    except Exception as e:
        print(f"Error in GUI: {e}")

if __name__ == "__main__":
    main()

