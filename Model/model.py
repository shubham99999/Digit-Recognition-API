import os
import sys
sys.path.append('../')
import csv
import joblib
import numpy as np
from declarations import ROOT_DIR
from Utility.ImageProcessing import imageToPixel,extract_number,show_image
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC

def train_model():
    
    # Reading data from train.csv file and storing it in a list named data. Each
    # element of "data" list is a row of train.csv file.
    data = []
    with open(os.path.join(ROOT_DIR,"static/Data/train.csv")) as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)

    # X - list of List to store 784 pixel values of each mnist image
    # Y - List to store digit which is present in the image
    X = []
    Y = []

    # Segment Image - i.e only keep black and white pixel
    # Hypothesis - Makes the model perform better as reduces the a lot values and still
    # retains the basic information about the shape of digit.
    # Same thing is also done with the image from the android application in the image processing module
    print("Performing Segmentation ....")

    for data_row in data[1:]:
        Z = lambda x: [255 if int(y)>0 else 0 for y in data_row[1:]]
        X.append(Z(data_row))
        Y.append([int(data_row[0])])

    # Converting X and Y to numpy arrays
    X = np.array(X)
    Y = np.array(Y)


    # Training set from the same distribution
    train_X = X[1:40000]
    train_Y = Y[1:40000]

    # Testing set from the same distribution
    test_X = X[40000:]
    test_Y = Y[40000:]

    # Create a model
    clf = RFC(n_estimators=50)

    print("Training Model...")

    # Fitting the model
    clf.fit(train_X,train_Y)

    # Store the model
    joblib.dump(clf,os.path.join(ROOT_DIR,'Model/model.pkl'))

    # Predict on test data from same distribution
    pred_Y = clf.predict(test_X)

    # Getting to know how our model performed on test data from mnist dataset.
    print("Metrics of trained model on MNIST TEST DATASET: ")
    print("Accuracy: ",accuracy_score(test_Y,pred_Y) * 100,"%")
    print("Confusion_matrix: ",confusion_matrix(test_Y,pred_Y))
    print("Recall: ",recall_score(test_Y,pred_Y,average='weighted'))
    print("Precision: ",precision_score(test_Y,pred_Y,average='weighted'))



def apply_model_on_test_set():
    pred_Y = []
    test_Y = []
    base_path = os.path.join(ROOT_DIR,"static/Data/Test Images")
    for f in os.listdir(base_path):
        test_Y.append(int(f.split('(')[0].strip()))
        extracted_numbers = extract_number(os.path.join(base_path,f),os.path.join(ROOT_DIR,"Model/intermediate/"))
        digits = extracted_numbers
        for idx in range(len(digits)):
            pixels = imageToPixel(digits[idx]).tolist()
            predicted_value = int(predict_value(pixels)[0])
            pred_Y.append(predicted_value)

    # Getting to know how our model performed on data from our android app. This is the data the model
    # will actually work with once it goes in production. That's why these metrics are much more important.
    print("Metrics of trained model on Images from Android Application (Our actual production data): ")
    print("Accuracy of trained model is: ",accuracy_score(test_Y,pred_Y) * 100,"%")
    print("Confusion_matrix: ",confusion_matrix(test_Y,pred_Y))
    print("Recall: ",recall_score(test_Y,pred_Y,average='weighted'))
    print("Precision: ",precision_score(test_Y,pred_Y,average='weighted'))


def predict_value(test_X):
    basepath = os.path.join(ROOT_DIR,"Model")

    # Load the already created model
    clf = joblib.load(os.path.join(basepath,"model.pkl"))

    # Predict on test data
    pred_Y = clf.predict(np.array(test_X))

    return clf.predict(test_X)

if __name__ == '__main__':
    train_model()
    apply_model_on_test_set()
