# ML classifier to translate face embeddings extracted from FaceNet
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import StackingClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn import svm
from random import choice
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot as plt
import datetime
from info_logging import log
import pickle
log("Script 2.2 Started at :: "+str(datetime.datetime.now()))

# load embeddings extracted via FaceNet in script 'mymodel2.2.py'
data = load('Image_DataSet_100_processed-embeddings.npz')
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
# fit model
#model = StackingClassifier(estimators=[('svc', svm.SVC(kernel = 'rbf', probability = True, random_state=0)), ('svr', make_pipeline(StandardScaler(), svm.SVC(kernel = 'linear', probability = True, random_state=0)))])
#model = BaggingClassifier(n_estimators=5, base_estimator=svm.SVC(), random_state=0)
#model = svm.SVC(kernel='rbf',random_state=0,probability=True)

model = StackingClassifier(estimators=[('svc1', svm.SVC(kernel = 'rbf', probability = True, random_state=0)),
                                       ('svc2', svm.SVC(kernel = 'rbf', probability = True, random_state=0)),
                                       ('svc3', svm.SVC(kernel = 'rbf', probability = True, random_state=0)),
                                       ('svc4', svm.SVC(kernel = 'rbf', probability = True, random_state=0)),
                                       ('svc5', svm.SVC(kernel = 'rbf', probability = True, random_state=0)),
                                       ('svc6', svm.SVC(kernel = 'rbf', probability = True, random_state=0)),
                             
                                       ('svr', make_pipeline(StandardScaler(), svm.SVC(kernel = 'rbf', probability = True, random_state=0)))])

model.fit(trainX, trainy)
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))


# predict output test label
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# accuracy score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
log('Train Accuracy Score :: '+str(score_train*100)+'% & Test Accuracy Score is :: '+ str(score_test*100)+'%')