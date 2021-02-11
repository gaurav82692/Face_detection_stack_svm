# face detection for the 200 Celebrity Faces Dataset
from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from info_logging import log
import datetime
log("Script Started at :: "+str(datetime.datetime.now()))

def face_cropper(file, detector, required_size=(125, 125)):
   try:
    image = Image.open(file)
    image = image.convert('RGB')
    pixels = asarray(image)
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array
   except:
    #Some images may not have faces so MTCNN may cause exception thus saving those images as it is
    log("File skipped from MTCNN :: "+str(file))
	 # load image from file
    image = Image.open(file)
	 # convert to RGB, if needed
    image = image.convert('RGB')
    image = image.resize(required_size)
	# convert to array
    face_array = asarray(image)
    return face_array


def face_loader(directory, detector):
	faces = list()	
	for filename in listdir(directory):
		path = directory + filename
		face = face_cropper(path, detector)
		faces.append(face)
	return faces


def load_data(dir):
	X, y = list(), list()
	# initializing MTCNN for face extraction using default weights
	detector = MTCNN()
	# enumerate folders, on per class
	for subdir in listdir(dir):
		path = dir + subdir + '/'
		if not isdir(path):
			continue
		faces = face_loader(path, detector)
		# create target labels
		labels = [subdir for _ in range(len(faces))]
		log("Loaded class :: "+str(subdir))
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)

def saving_files(folder_path):
    # loading training dataset
    trainX, trainy = load_data(folder_path+'\\train\\')
    log(str(trainX.shape)+" "+str(trainy.shape))
    # loading test dataset
    testX, testy = load_data(folder_path+'\\test\\')
    # saving data to compressed format
    savez_compressed('Image_DataSet_100_Processed.npz', trainX, trainy, testX, testy)
    log("Script Ended at :: "+str(datetime.datetime.now()))

if __name__ == '__main__':
    folder_path='processed_dataset'
    saving_files(folder_path)