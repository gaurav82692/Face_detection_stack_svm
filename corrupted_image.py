from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn import svm
from random import choice
from numpy import load
from numpy import expand_dims
import numpy as np
from matplotlib import pyplot as plt
import datetime
from info_logging import log
from PIL import Image
import pickle
import os

#load model
filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

data = load('Image_DataSet_100_Processed-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
log('Shape of training data :: '+str(trainX.shape[0])+" & shape of test data :: "+ str(testX.shape[0]))
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

#load faces train
data = load('Image_DataSet_100_Processed.npz')
trainX_faces = data['arr_0']

os.mkdir('Corrupted Images')

for selection in range(trainX.shape[0]):
    #selection = choice([i for i in range(trainX.shape[0])])
    print(selection)
    random_face_pixels = trainX_faces[selection]
    random_face_emb = trainX[selection]
    random_face_class = trainy[selection]
    random_face_name = out_encoder.inverse_transform([random_face_class])
    # prediction for the face
    samples = expand_dims(random_face_emb, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)
    # get name
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    #log('Predicted :: '+str (predict_names[0]) + ' Expected :: '+str (random_face_name[0]))
    if (predict_names[0] != random_face_name[0]):
        im = Image.fromarray(random_face_pixels.astype(np.uint8));
        file_name = 'Corrupted Images\\' + random_face_name[0]+str(selection)+"tr.jpg"
        log(str(file_name))
        im.save(file_name) 
		
log("Script 2.2 Ended at :: "+str(datetime.datetime.now()))

# load faces test
data = load('Image_DataSet_100_processed.npz')
testX_faces = data['arr_2']

for selection in range(testX.shape[0]):
    #selection = choice([i for i in range(testX.shape[0])])
    print(selection)
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
    #log('Predicted :: '+str (predict_names[0]) + ' Expected :: '+str (random_face_name[0]))
    if (predict_names[0] != random_face_name[0]):
        im = Image.fromarray(random_face_pixels.astype(np.uint8))
        file_name = 'Corrupted Images\\' +random_face_name[0]+str(selection)+"ts.jpg"
        log(str(file_name))
        im.save(file_name)

log("Script 2.2 Ended at :: "+str(datetime.datetime.now()))
