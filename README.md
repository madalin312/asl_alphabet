# asl_alphabet

ASL Letter Recognition Using Deep Neural Networks

This repository contains a project that predicts American Sign Language (ASL) letters using a deep neural network (DNN). The project utilizes the Mediapipe Hands (mphands) module to detect the location of hands in an image and performs inference on the selected rectangle where the hands are located.

Project Overview
The goal of this project is to develop a system that can recognize and predict the ASL letter being shown by a hand in an image. The process involves detecting the hand using mphands, cropping the region of interest (ROI), and feeding it into a deep neural network that performs the classification.

Dataset
The dataset used for training was built by capturing images of hands performing ASL letters. The process involved:

Hand Detection: Using the mphands module, the location of the hand was detected in each image.
Region of Interest (ROI): The detected hand area was cropped to create the ROI, which focuses on the hand making the ASL gesture.
Data Collection: Images were captured for each ASL letter (A-Z), and the cropped hand images were saved as the dataset.
Data Structure
The dataset is structured as follows:

asl_alphabet_dataset/
│
├── A/
│   ├── A1.jpg
│   ├── A2.jpg
│   └── ...
├── B/
│   ├── B1.jpg
│   ├── B2.jpg
│   └── ...
└── ...

Each folder represents a different ASL letter, containing images of hands making that letter.

Training Process
The model was trained on the dataset of hand images, using the following steps:

Data Augmentation: Applied to the training images to improve generalization.
Loss Function: Categorical Cross-Entropy.
Optimizer: Adam optimizer with a learning rate of 0.001.
Evaluation: The model was evaluated using accuracy, precision, recall, and confusion matrix.
Inference Pipeline
The inference process involves the following steps:

Hand Detection: mphands detects the location of the hand in a real-time video or static image.
ROI Extraction: The detected hand region is cropped to form the ROI.
Model Inference: The ROI is passed through the trained DNN model to predict the ASL letter.
Result Display: The predicted letter is displayed on the screen.
How to Run the Project
Prerequisites
Python 3.x

Required Python libraries (listed in requirements.txt):
mediapipe
tensorflow
opencv-python
numpy
matplotlib
Installation
Clone the repository:


git clone https://github.com/madalin312/asl_alphabet
cd asl-letter-recognition
python asl_letters.py