#FaceNet is a system that, given a picture of a face, will extract high-quality
#features from the face and predict a 128 element vector representation these features, called a face embedding.
#this script will execute faceNet
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
from info_logging import log
import time
import datetime

log("Script 2.1 Started at :: "+str(datetime.datetime.now()))
# get the face embedding for one face
def embedding_generator(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

# loading the face dataset which came as output from script named as 'mymodel2.1.py'
data = load('Image_DataSet_100_Processed.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
log('Loading Train/Test Data '+str(trainX.shape)+":: "+str(trainy.shape)+" :: "+str(testX.shape)+" :: "+str(testy.shape))
# load the facenet model
model = load_model('embedding_gen_model.h5')
log('Model Loaded Successfuly')
# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
	embedding = embedding_generator(model, face_pixels)
	newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
log(str(newTrainX.shape))
# converting faces to embeddings
newTestX = list()
for face_pixels in testX:
	embedding = embedding_generator(model, face_pixels)
	newTestX.append(embedding)
newTestX = asarray(newTestX)
log('Test Data :: '+str(newTestX.shape))
savez_compressed('Image_DataSet_100_Processed-embeddings.npz', newTrainX, trainy, newTestX, testy)
log("Script 2.1 Ended at :: "+str(datetime.datetime.now()))