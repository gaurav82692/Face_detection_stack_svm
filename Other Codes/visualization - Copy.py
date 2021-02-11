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
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        
        ax = fig.add_subplot(2, 5, idx + 1, xticks=[], yticks=[])
        idx += 1
        plt.imshow(random_face_pixels)
        ax.set_title("\n\nPredicted Name: {} \n Actual Name: {} \n probability: {}".format(predict_names[0], random_face_name[0], class_probability),
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
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        y_pred.append(predict_names)

    return y_true, y_pred, labels

if __name__ == '__main__':
    #load model
    filename = 'finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))
    y_true, y_pred, label = test(model)
    score = accuracy_score(y_true, y_pred)
    log("Accuracy Score on test data: " + str(score))
    cm = confusion_matrix(y_true, y_pred, labels=label)
    np.savetxt("confusion_mat.csv", cm, delimiter=",")