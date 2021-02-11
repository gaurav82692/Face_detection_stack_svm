import cv2
import numpy as np
from info_logging import log
import pandas as pd
import imageio
import os

def sharpen():
    log("####################### Image Sharpening started ######################")
    #processed_dir is input folder on which sharpening needs    to be done
    processed_dir = 'C:\\Final Project\\Corrected Images'
    #root_dir is the output folder for sharpened images
    root_dir = 'C:\\Final Project\\Images Sharpen'

    #create directory structure for saving cropped images
    os.mkdir(root_dir)
    classes_dir = os.listdir(processed_dir)
  
    error = 0

    ################Processing Started for Sharpening face images######################
    log("Sharpening Script started at :: ")
    for cls in classes_dir:
        src = processed_dir + "\\" + cls  # Folder to copy images from
        path = root_dir + '\\' + cls
        os.makedirs(root_dir + '\\' + cls)

        # Copy-pasting images
        allFileNames = os.listdir(src)
        for name in allFileNames:
            img_path = src + '\\' + name
            final_path = path + '\\' + name
            img = cv2.imread(img_path)
        # defining sharpening kernel for images
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpen = cv2.filter2D(img, -1, sharpen_kernel)
            cv2.imwrite(final_path, sharpen)

    log("Script Ended at :: ")
    log("###################### images Successfully Sharpened and saved at Images Sharpen Folder ###################")


if __name__ == '__main__':
    sharpen()
