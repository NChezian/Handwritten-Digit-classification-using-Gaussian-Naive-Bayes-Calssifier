import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

# Input the MNIST dataset in the form of .CSV files and read them using  Pandas library
train_data = pd.read_csv("C:/Users/arunk/OneDrive/Desktop/ICT_Nik/mnist_train.csv")
test_data = pd.read_csv("C:/Users/arunk/OneDrive/Desktop/ICT_Nik/mnist_test.csv")

# Splitting the given the data into features(X) and labels(y)
train_features = train_data.values[:, 1:]
train_labels = train_data.values[:, 0]
test_features = test_data.values[:, 1:]
test_labels = test_data.values[:, 0]

# Using the GaussianNB func. to generate Gaussian Naive Bayes model
gnb = GaussianNB()

# Next step is to train the model based on the training data
gnb.fit(train_features,train_labels)

# Now predict the label of test data using the trained model
y_pred = gnb.predict(test_features)

# Generate the confusion matrix and evaluate the model
cm = confusion_matrix(test_labels, y_pred)
print("Confusion Matrix:\n", cm)

# Evaluate the model using Precision, Recall, and F1-score
precision, recall, f1, support = precision_recall_fscore_support(test_labels, y_pred)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Evaluate the model using Accuracy
accuracy = accuracy_score(test_labels, y_pred)
print("Accuracy:", accuracy)
