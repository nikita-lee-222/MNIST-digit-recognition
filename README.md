# MNIST Digit Recognition Web Application

This project is a Flask-based web application for handwritten digit recognition (0–9). Trained on the MNIST dataset using TensorFlow/Keras.  
Users can draw a digit in the browser and the application will predict the number along with a confidence score.

## Features
- Convolutional Neural Network (CNN) built with TensorFlow/Keras
- Flask server with a JSON endpoint
- Web interface to draw digits
- Support for model training from scratch or using a pre-trained model

## Pre-trained Model
To save time, you can download a pre-trained model (`mnist_model.h5`) from Google Drive:  
[Download Pre-trained Model](https://drive.google.com/drive/folders/19h4i68wfja4xEvLxRIThsvCONV5UfoyR?usp=sharing)  

You can also train the model yourself using the included training script (`train_model.py`).  
For faster training, you can adjust the input size from **280x280** back to the original **28x28** used in MNIST.

## How to use?

### 1. Clone the repository
```bash
git clone https://github.com/nikita-lee-222/MNIST-digit-recognition
cd MNIST-digit-recognition
```
- Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   #Linux/Mac
venv\Scripts\activate      #WinOS
```
- Install dependencies
```bash
pip install -r requirements.txt
```
- (Option A) Train the model
```bash
python train.py #You can also make a test of your trained moddel. Just run test_model.py
```
- (Option B) Use a pre-trained model
```bash
model/mnist_model.h5
```
- Run the server
```bash
python app.py
```
The application will be available at:
http://127.0.0.1:5000/

This project was developed with guidance and reference materials from online resources and tools.
