# Analog Clock Time Recognition Model

This repository contains the code to train and evaluate a deep learning model that recognizes the time from images of analog clock faces. The model is built using a convolutional neural network (CNN) to process grayscale images and predict the time based on the positions of the clock's hands.

## Dataset

The dataset consists of monochrome images of analog clocks, with filenames indicating the time shown on the clock. The images are preprocessed and the labels are extracted directly from the filenames.

- **Dataset Path:** `https://www.kaggle.com/datasets/kopfgeldjaeger/simple-analog-clock-monochrome`
- **Labels:** The labels are in the format of minutes past midnight, encoded as one-hot vectors.

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following structure:

1. **Input Layer:** 224x224 grayscale images.
2. **Convolutional Layers:** 
   - 5 convolutional layers with ReLU activation and Batch Normalization.
   - MaxPooling layers after each convolutional block to reduce spatial dimensions.
3. **Fully Connected Layers:** 
   - 4 fully connected layers with ReLU activation and Dropout for regularization.
4. **Output Layer:** 
   - A softmax layer with `720` units, representing each minute in a 12-hour period.

The model is compiled with the Adam optimizer and categorical cross-entropy loss. It is trained for up to 100 epochs with early stopping to prevent overfitting.

## Technologies Used

1. Python 3.x
2. TensorFlow/Keras
3. OpenCV
4. NumPy
5. Matplotlib
