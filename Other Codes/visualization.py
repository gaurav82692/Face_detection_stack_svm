from numpy import load
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn import svm
from random import choice
from numpy import load
from numpy import expand_dims
import numpy as np
from matplotlib import pyplot as plt
import datetime
from info_logging import log
import pickle
import pandas as pd

def test(model):
    labels = []
    y_true = []
    y_pred = []

    data = load('Image_DataSet_100_processed-embeddings.npz')
    trainy, testX, testy = data['arr_1'], data['arr_2'], data['arr_3']
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    testX = in_encoder.transform(testX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    testy = out_encoder.transform(testy)

    # load faces test
    data = load('Image_DataSet_100_processed.npz')
    testX_faces = data['arr_2']

    fig = plt.figure(figsize=(25, 10))
    idx = 0
    for itr in range(10):
        selection = choice([i for i in range(testX.shape[0])])
        random_face_pixels = testX_faces[selection]
        random_face_emb = testX[selection]
        random_face_class = testy[selection]
        random_face_name = out_encoder.inverse_transform([random_face_class])
        # prediction for the face
        samples = expand_dims(random_face_emb, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)
        
        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index]
        predict_names = out_encoder.inverse_transform(yhat_class)
        
        ax = fig.add_subplot(2, 5, idx + 1, xticks=[], yticks=[])
        idx += 1
        plt.imshow(random_face_pixels)
        ax.set_title("\n\nPredicted Name: {} \n Actual Name: {} \n probability: {}".format(predict_names[0], random_face_name[0], round(class_probability, 4)),
                     color=("green" if predict_names[0] == random_face_name[0] else "red"))
    fig.savefig('test_visualization.jpg')
    log("Visualized results can be found in test_visualization.jpg file")

    for selection in range(testy.shape[0]):
        random_face_pixels = testX_faces[selection]
        random_face_emb = testX[selection]
        random_face_class = testy[selection]
        random_face_name = out_encoder.inverse_transform([random_face_class])
        # prediction for the face
        samples = expand_dims(random_face_emb, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)

        if random_face_name[0] not in labels:
            labels.append(random_face_name[0])
        y_true.append(random_face_name[0])
        
        # get name
        predict_names = out_encoder.inverse_transform(yhat_class)
        y_pred.append(predict_names)

    return y_true, y_pred, labels

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          normalize = False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    import itertools
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=2)
    plt.yticks(tick_marks, classes, fontsize=2)

    #Uncomment the following lines if you also want to include exact numbers in Confusion matrix
    """    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    """
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("Confusion_matrix_100.jpg")
    log("Confusion matrix saved as Confusion_matrix.jpg")

if __name__ == '__main__':
    #load model
    filename = 'finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))
    y_true, y_pred, labels = test(model)
    score = accuracy_score(y_true, y_pred)
    log("Accuracy Score on test data: " + str(score))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt("confusion_mat.csv", cm, delimiter=",")
    plot_confusion_matrix(cm, classes=labels)