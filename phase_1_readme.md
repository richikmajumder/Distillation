# Bayesian Optimized 1-bit CNN (BONN) for CIFAR-10

This project implements a Bayesian Optimized 1-bit Convolutional Neural Network (BONN) using Wide-ResNet as the backbone. The model is trained on the CIFAR-10 dataset.

## Prerequisites

Before running the project, make sure you have Python 3.7+ installed. You can install the required dependencies using the following command:

pip install -r requirements.txt

Running the Code
Download CIFAR-10 dataset: The dataset is automatically downloaded when running the script.

Training the model: The following command will train the model on CIFAR-10 and display the training progress:

python train_test_analysis.py
The model will train for 200 epochs by default.
During each epoch, the loss and accuracy for both training and testing phases will be displayed.
View results: Once the training is complete, a plot showing the loss and accuracy over the epochs will be saved as training_metrics.png and displayed.

Files
train_test_analysis.py: The main script containing the code for training, testing, and analyzing the model.
requirements.txt: File listing the Python dependencies required to run the project.

Make sure to adjust the number of epochs or batch size accordingly.
