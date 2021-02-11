from sklearn.preprocessing import Normalizer, LabelEncoder
from PIL import Image
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
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
from info_logging import log
from numpy import asarray

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename_of_image = upload.filename
        destination = "/".join([target, filename_of_image])
        print("Accept incoming file:", filename_of_image)
        print("Save it to:", destination)
        upload.save(destination)
    folder = 'images'
    ex = folder+'/'+filename_of_image
    data = load('static/imgset.npz')
    print(data)
    labels = data['arr_1']

    # image preprocessing
    imgactual = np.asarray(Image.open(ex))
    img = Ying_2017_CAIP(imgactual)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    im = Image.fromarray(sharpen)
    im.save('sharpen.jpg')
    #img = face_cropper('sharpen.jpg', detector = MTCNN())
    #############################################################################-------face_cropper-----##################
    required_size = (125, 125)
    file = 'sharpen.jpg'
    detector = MTCNN()
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
        img = face_array
    except:
        # Some images may not have faces so MTCNN
        # may cause exception thus saving those images as it is
        log("File skipped from MTCNN :: "+str(file))
        # load image from file
        image = Image.open(file)
        # convert to RGB, if needed
        image = image.convert('RGB')
        image = image.resize(required_size)
        # convert to array
        face_array = asarray(image)
        img = face_array
    ############################################################################----end_face_cropper----###################
    if os.path.exists("sharpen.jpg"):
        os.remove("sharpen.jpg")

    # generating embeddings
    modeldl = load_model('facenet_keras.h5')
    #embeddings = embedding_generator(modeldl, img)
    ###########################################################################----------embedding---------#################
    model = modeldl
    face_pixels = img
    face_pixels = face_pixels.astype('float32')
    mean = face_pixels.mean()
    std = face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    embeddings = yhat[0]
    ###########################################################################--------end_embedding-------#################
    print(embeddings)
    embed = expand_dims(embeddings, axis=0)

    # normalizing image defining the Label encoder
    in_encoder = Normalizer(norm='l2')
    embed = in_encoder.transform(embed)
    out_encoder = LabelEncoder()
    out_encoder.fit(labels)

    # predicting final output
    filename = 'finalized_model.sav'
    modelf = pickle.load(open(filename, 'rb'))
    yhat_class = modelf.predict(embed)
    yhat_prob = modelf.predict_proba(embed)
    thresh=np.amax(yhat_prob)
    # get name
    predict_names = out_encoder.inverse_transform(yhat_class)

    print('Predicted Name: '+str(predict_names))
    # visualizing prediction
    plt.imshow(imgactual)
    plt.title('Predicted Name: '+str(predict_names[0]))
    plt.show()
    if thresh>0.8:
        return render_template("complete_display_image.html", image_name=filename_of_image, pred_name=str(predict_names))
    else:
        un="Unknown"
        return render_template("complete_display_image.html", image_name=filename_of_image, pred_name=str(un))



@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)


@app.route('/go back')
def back():
    return render_template("upload.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4466, debug=True)
