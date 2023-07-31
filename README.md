Gaussian Naive Bayes for MNIST Classification
This repository contains a Python script for implementing Gaussian Naive Bayes on the MNIST dataset, 
a widely used benchmark dataset for handwritten digit classification. The code utilizes the scikit-learn 
library to train and evaluate the model.

Requirements
Python 3.x
pandas
scikit-learn
Usage
Clone this repository to your local machine.
Make sure you have the required libraries installed by running pip install -r requirements.txt.
Download the MNIST dataset in CSV format from the provided links and adjust the file paths in the code accordingly.
Execute the script to train the Gaussian Naive Bayes model and evaluate its performance on the test data.
Description
The code reads the MNIST dataset in the form of CSV files using the Pandas library. It then splits the data into features 
(X) and labels (y). The GaussianNB function from scikit-learn is used to create and train the Gaussian Naive Bayes model. 
The model is then used to predict the labels of the test data, and the script evaluates its performance using the confusion matrix, 
precision, recall, F1-score, and accuracy.

Output
The script will display the following metrics:

Confusion Matrix: A matrix representing the true positive, false positive, true negative, and false negative predictions.
Precision: The precision score for each class.
Recall: The recall score for each class.
F1-score: The F1-score for each class.
Accuracy: The overall accuracy of the model.
Acknowledgments
This code is built upon the functionalities of the Pandas and scikit-learn libraries. 
The MNIST dataset is widely used for benchmarking machine learning algorithms.

Feel free to use this code for your MNIST digit classification tasks. If you find it helpful, kindly consider giving credit to this repository.

Note: This code is meant for educational and experimental purposes, and further optimization may be required for large-scale datasets or real-world applications.
