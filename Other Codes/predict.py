from sklearn.preprocessing import Normalizer, LabelEncoder
from PIL import Image
from numpy import load, expand_dims
#from mymodel2 import face_cropper
#from mymodel3 import embedding_generator
from keras.models import load_model
from Pipeline.image_corrector import Ying_2017_CAIP
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os

def embedding_generator1(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]
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

def predict(img_path):
    data = load('Image_DataSet_100_processed-embeddings.npz')
    labels = data['arr_1']
 
    #image preprocessing
    imgactual = np.asarray(Image.open(img_path))
    img = Ying_2017_CAIP(imgactual)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    im = Image.fromarray(sharpen)
    im.save('sharpen.jpg')
    img = face_cropper('sharpen.jpg', detector = MTCNN())
    if os.path.exists("sharpen.jpg"):
        os.remove("sharpen.jpg")

    #generating embeddings
    modeldl = load_model('facenet_keras.h5')
    embeddings = embedding_generator1(modeldl, img)
    print(embeddings)
    embed = expand_dims(embeddings, axis=0)

    #normalizing image defining the Label encoder
    in_encoder = Normalizer(norm='l2')
    embed = in_encoder.transform(embed)
    out_encoder = LabelEncoder()
    out_encoder.fit(labels)

    #predicting final output
    filename = 'finalized_model.sav'
    modelf = pickle.load(open(filename, 'rb'))
    yhat_class = modelf.predict(embed)
    yhat_prob = modelf.predict_proba(embed) 
    # get name
    predict_names = out_encoder.inverse_transform(yhat_class)

    print('Predicted Name: '+str(predict_names))
    #visualizing prediction
    plt.imshow(imgactual)
    plt.title('Predicted Name: '+str(predict_names[0]))
    plt.show()

if __name__ == '__main__':
    predict('1.jpg')

    
    
